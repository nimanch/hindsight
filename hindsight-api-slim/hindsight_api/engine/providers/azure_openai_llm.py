"""
Azure OpenAI LLM provider using AsyncAzureOpenAI SDK.

This provider handles Azure AI Foundry (Azure OpenAI) endpoints with:
- Entra ID authentication via DefaultAzureCredential (default, P0)
- API Key authentication (fallback)
- Native azure_endpoint and api_version handling
- Deployment name mapping via model parameter
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any

from openai import APIConnectionError, APIStatusError, AsyncAzureOpenAI, LengthFinishReasonError

from hindsight_api.config import DEFAULT_LLM_TIMEOUT, ENV_LLM_TIMEOUT
from hindsight_api.engine.llm_interface import LLMInterface, OutputTooLongError
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult, TokenUsage
from hindsight_api.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


def _strip_code_fences(content: str) -> str:
    """Strip markdown code fences from LLM response if present."""
    if "```" not in content:
        return content
    try:
        if "```json" in content:
            return content.split("```json")[1].split("```")[0].strip()
        return content.split("```")[1].split("```")[0].strip()
    except (IndexError, ValueError):
        return content


class AzureOpenAILLM(LLMInterface):
    """
    Azure OpenAI provider using AsyncAzureOpenAI SDK.

    Supports:
    - Azure AI Foundry endpoints (cognitiveservices.azure.com)
    - Entra ID auth via DefaultAzureCredential (P0, default)
    - API Key auth (fallback when use_entra_id=False)
    - Deployment name mapping via model parameter
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        reasoning_effort: str = "low",
        timeout: float | None = None,
        azure_endpoint: str | None = None,
        azure_api_version: str = "2024-12-01-preview",
        azure_deployment_name: str | None = None,
        azure_use_entra_id: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize Azure OpenAI LLM provider.

        Args:
            provider: Provider name ("azure").
            api_key: API key (used when azure_use_entra_id=False).
            base_url: Base URL (used as azure_endpoint fallback).
            model: Model/deployment name (e.g., "gpt-4o").
            reasoning_effort: Reasoning effort for supported models.
            timeout: Request timeout in seconds.
            azure_endpoint: Azure endpoint URL (e.g., https://myresource.cognitiveservices.azure.com/).
            azure_api_version: Azure OpenAI API version.
            azure_deployment_name: Explicit deployment name (defaults to model).
            azure_use_entra_id: If True, use DefaultAzureCredential for auth.
            **kwargs: Additional parameters.
        """
        super().__init__(provider, api_key, base_url, model, reasoning_effort, **kwargs)

        self.azure_endpoint = azure_endpoint or base_url
        if not self.azure_endpoint:
            raise ValueError(
                "Azure endpoint is required. Set HINDSIGHT_API_LLM_AZURE_ENDPOINT or HINDSIGHT_API_LLM_BASE_URL."
            )

        self.azure_api_version = azure_api_version
        self.deployment_name = azure_deployment_name or model
        self.azure_use_entra_id = azure_use_entra_id
        self.timeout = timeout or float(os.getenv(ENV_LLM_TIMEOUT, str(DEFAULT_LLM_TIMEOUT)))

        # Build AsyncAzureOpenAI client
        client_kwargs: dict[str, Any] = {
            "api_version": self.azure_api_version,
            "azure_endpoint": self.azure_endpoint,
            "max_retries": 0,
        }

        if self.timeout:
            client_kwargs["timeout"] = self.timeout

        if self.azure_use_entra_id:
            try:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider

                credential = DefaultAzureCredential()
                token_provider = get_bearer_token_provider(
                    credential, "https://cognitiveservices.azure.com/.default"
                )
                client_kwargs["azure_ad_token_provider"] = token_provider
                logger.info("Azure OpenAI: Using Entra ID authentication (DefaultAzureCredential)")
            except ImportError:
                raise ImportError(
                    "azure-identity package is required for Entra ID authentication. "
                    "Install it with: pip install azure-identity"
                )
        else:
            if not self.api_key:
                raise ValueError("API key is required when Entra ID auth is disabled (azure_use_entra_id=False)")
            client_kwargs["api_key"] = self.api_key
            logger.info("Azure OpenAI: Using API Key authentication")

        self._client = AsyncAzureOpenAI(**client_kwargs)
        logger.info(
            f"Azure OpenAI client initialized: endpoint={self.azure_endpoint}, "
            f"deployment={self.deployment_name}, api_version={self.azure_api_version}"
        )

    async def verify_connection(self) -> None:
        """Verify Azure OpenAI connectivity with a simple test call."""
        try:
            logger.info(f"Verifying Azure OpenAI connection: {self.azure_endpoint}/{self.deployment_name}")
            await self.call(
                messages=[{"role": "user", "content": "Say 'ok'"}],
                max_completion_tokens=100,
                max_retries=2,
                initial_backoff=0.5,
                max_backoff=2.0,
                scope="verification",
            )
            logger.info(f"Azure OpenAI connection verified: {self.deployment_name}")
        except Exception as e:
            raise RuntimeError(
                f"Azure OpenAI connection verification failed for {self.azure_endpoint}/{self.deployment_name}: {e}"
            ) from e

    def _supports_reasoning_model(self) -> bool:
        """Check if the current model is a reasoning model."""
        model_lower = self.model.lower()
        return any(x in model_lower for x in ["gpt-5", "o1", "o3", "deepseek"])

    async def call(
        self,
        messages: list[dict[str, str]],
        response_format: Any | None = None,
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        scope: str = "memory",
        max_retries: int = 10,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        skip_validation: bool = False,
        strict_schema: bool = False,
        return_usage: bool = False,
    ) -> Any:
        """Make an Azure OpenAI API call with retry logic."""
        start_time = time.time()

        call_params: dict[str, Any] = {
            "model": self.deployment_name,
            "messages": messages,
        }

        is_reasoning_model = self._supports_reasoning_model()

        if max_completion_tokens is not None:
            if is_reasoning_model and max_completion_tokens < 16000:
                max_completion_tokens = 16000
            call_params["max_completion_tokens"] = max_completion_tokens

        if temperature is not None and not is_reasoning_model:
            call_params["temperature"] = temperature

        if is_reasoning_model:
            call_params["reasoning_effort"] = self.reasoning_effort

        # Prepare response format
        if response_format is not None:
            schema = None
            if hasattr(response_format, "model_json_schema"):
                schema = response_format.model_json_schema()

            if strict_schema and schema is not None:
                call_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "strict": True,
                        "schema": schema,
                    },
                }
            else:
                if schema is not None:
                    schema_msg = (
                        f"\n\nYou must respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
                    )
                    if call_params["messages"] and call_params["messages"][0].get("role") == "system":
                        first_msg = call_params["messages"][0]
                        if isinstance(first_msg, dict) and isinstance(first_msg.get("content"), str):
                            first_msg["content"] += schema_msg
                    elif call_params["messages"]:
                        first_msg = call_params["messages"][0]
                        if isinstance(first_msg, dict) and isinstance(first_msg.get("content"), str):
                            first_msg["content"] = schema_msg + "\n\n" + first_msg["content"]
                call_params["response_format"] = {"type": "json_object"}

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                response = await self._client.chat.completions.create(**call_params)

                if response_format is not None:
                    content = response.choices[0].message.content

                    if content:
                        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
                        content = re.sub(r"<thinking>.*?</thinking>", "", content, flags=re.DOTALL)
                        content = re.sub(r"<reasoning>.*?</reasoning>", "", content, flags=re.DOTALL)
                        content = content.strip()

                    clean_content = _strip_code_fences(content)
                    try:
                        json_data = json.loads(clean_content)
                    except json.JSONDecodeError:
                        try:
                            json_data = json.loads(content)
                        except json.JSONDecodeError as json_err:
                            content_preview = content[:500] if content else "<empty>"
                            logger.warning(
                                f"Azure OpenAI JSON parse error (attempt {attempt + 1}/{max_retries + 1}): {json_err}\n"
                                f"  Deployment: {self.deployment_name}\n"
                                f"  Content preview: {content_preview!r}"
                            )
                            if attempt < max_retries:
                                backoff = min(initial_backoff * (2**attempt), max_backoff)
                                await asyncio.sleep(backoff)
                                last_exception = json_err
                                continue
                            raise

                    result = json_data if skip_validation else response_format.model_validate(json_data)
                else:
                    response = await self._client.chat.completions.create(**call_params)
                    result = response.choices[0].message.content

                # Record metrics
                duration = time.time() - start_time
                usage = response.usage
                input_tokens = usage.prompt_tokens or 0 if usage else 0
                output_tokens = usage.completion_tokens or 0 if usage else 0
                total_tokens = usage.total_tokens or 0 if usage else 0

                metrics = get_metrics_collector()
                metrics.record_llm_call(
                    provider=self.provider,
                    model=self.deployment_name,
                    scope=scope,
                    duration=duration,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=True,
                )

                try:
                    from hindsight_api.tracing import get_span_recorder
                    span_recorder = get_span_recorder()
                    response_str = result if isinstance(result, str) else json.dumps(result) if isinstance(result, dict) else str(result)
                    span_recorder.record_llm_call(
                        provider=self.provider, model=self.deployment_name, scope=scope,
                        messages=messages, response_content=response_str,
                        input_tokens=input_tokens, output_tokens=output_tokens,
                        duration=duration,
                        finish_reason=response.choices[0].finish_reason if response.choices else None,
                        error=None,
                    )
                except Exception:
                    pass

                if return_usage:
                    return result, TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens)
                return result

            except LengthFinishReasonError as e:
                raise OutputTooLongError(str(e)) from e

            except APIConnectionError as e:
                last_exception = e
                if attempt < max_retries:
                    backoff = min(initial_backoff * (2**attempt), max_backoff)
                    logger.warning(f"Azure OpenAI connection error (attempt {attempt + 1}): {e}. Retrying in {backoff}s")
                    await asyncio.sleep(backoff)
                    continue
                raise

            except APIStatusError as e:
                if e.status_code in (401, 403):
                    logger.error(f"Azure OpenAI auth error: {e}. Check Entra ID credentials or API key.")
                    raise
                last_exception = e
                if attempt < max_retries:
                    backoff = min(initial_backoff * (2**attempt), max_backoff)
                    retry_after = None
                    if hasattr(e, "response") and e.response is not None:
                        retry_after_val = e.response.headers.get("retry-after-ms")
                        if retry_after_val:
                            retry_after = float(retry_after_val) / 1000.0
                        else:
                            retry_after_val = e.response.headers.get("retry-after")
                            if retry_after_val:
                                retry_after = float(retry_after_val)
                    actual_backoff = retry_after if retry_after else backoff
                    logger.warning(f"Azure OpenAI API error {e.status_code} (attempt {attempt + 1}): {e}. Retrying in {actual_backoff}s")
                    await asyncio.sleep(actual_backoff)
                    continue
                raise

            except Exception:
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Azure OpenAI call failed after all retries")

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        scope: str = "tools",
        max_retries: int = 5,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> LLMToolCallResult:
        """Make an Azure OpenAI API call with tool/function calling support."""
        start_time = time.time()

        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            forced_name = tool_choice.get("function", {}).get("name")
            if forced_name:
                filtered = [t for t in tools if t.get("function", {}).get("name") == forced_name]
                if filtered:
                    tools = filtered
                tool_choice = "required"

        call_params: dict[str, Any] = {
            "model": self.deployment_name,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
        }

        if max_completion_tokens is not None:
            call_params["max_completion_tokens"] = max_completion_tokens
        if temperature is not None:
            call_params["temperature"] = temperature

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                response = await self._client.chat.completions.create(**call_params)
                message = response.choices[0].message
                finish_reason = response.choices[0].finish_reason

                tool_calls: list[LLMToolCall] = []
                if message.tool_calls:
                    for tc in message.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                        except json.JSONDecodeError:
                            args = {"_raw": tc.function.arguments}
                        tool_calls.append(LLMToolCall(id=tc.id, name=tc.function.name, arguments=args))

                content = message.content
                duration = time.time() - start_time
                usage = response.usage
                input_tokens = usage.prompt_tokens or 0 if usage else 0
                output_tokens = usage.completion_tokens or 0 if usage else 0

                metrics = get_metrics_collector()
                metrics.record_llm_call(
                    provider=self.provider, model=self.deployment_name, scope=scope,
                    duration=duration, input_tokens=input_tokens, output_tokens=output_tokens, success=True,
                )

                try:
                    from hindsight_api.tracing import get_span_recorder
                    span_recorder = get_span_recorder()
                    tool_calls_dict = [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls] if tool_calls else None
                    span_recorder.record_llm_call(
                        provider=self.provider, model=self.deployment_name, scope=scope,
                        messages=messages, response_content=content,
                        input_tokens=input_tokens, output_tokens=output_tokens,
                        duration=duration, finish_reason=finish_reason, error=None, tool_calls=tool_calls_dict,
                    )
                except Exception:
                    pass

                return LLMToolCallResult(
                    content=content, tool_calls=tool_calls, finish_reason=finish_reason,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                )

            except APIConnectionError as e:
                last_exception = e
                if attempt < max_retries:
                    await asyncio.sleep(min(initial_backoff * (2**attempt), max_backoff))
                    continue
                raise

            except APIStatusError as e:
                if e.status_code in (401, 403):
                    raise
                last_exception = e
                if attempt < max_retries:
                    backoff = min(initial_backoff * (2**attempt), max_backoff)
                    retry_after = None
                    if hasattr(e, "response") and e.response is not None:
                        retry_after_val = e.response.headers.get("retry-after-ms")
                        if retry_after_val:
                            retry_after = float(retry_after_val) / 1000.0
                    actual_backoff = retry_after if retry_after else backoff
                    await asyncio.sleep(actual_backoff)
                    continue
                raise

            except Exception:
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Azure OpenAI tool call failed after all retries")

    async def cleanup(self) -> None:
        """Clean up the Azure OpenAI client."""
        if hasattr(self, "_client") and self._client:
            await self._client.close()
