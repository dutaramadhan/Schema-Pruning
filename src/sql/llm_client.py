from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMConfig:
    name: str
    backend: str
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 512
    system_prompt: Optional[str] = None
    timeout_s: float = 60.0
    predictions_path: Optional[str] = None


class LLMClient:
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(LLMClient):
    def __init__(self, config: LLMConfig) -> None:
        if not config.base_url:
            raise ValueError("base_url is required for openai_compatible backend.")
        if not config.model:
            raise ValueError("model is required for openai_compatible backend.")
        self.base_url = config.base_url.rstrip("/")
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.timeout_s = config.timeout_s
        self.api_key = None
        if config.api_key_env:
            self.api_key = os.environ.get(config.api_key_env)
        self._headers = {"Content-Type": "application/json"}
        if self.api_key:
            self._headers["Authorization"] = f"Bearer {self.api_key}"

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers, method="POST")
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore") if exc.fp else str(exc)
            raise RuntimeError(f"LLM request failed: {detail}") from exc
        elapsed = time.perf_counter() - start
        payload_out = json.loads(raw)
        content = _extract_openai_content(payload_out)
        if not content:
            raise RuntimeError(f"LLM returned empty response after {elapsed:.2f}s.")
        return content.strip()


def _extract_openai_content(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    return ""


def build_llm_client(config: LLMConfig) -> Optional[LLMClient]:
    if config.backend == "openai_compatible":
        return OpenAICompatibleClient(config)
    if config.backend == "predictions_file":
        return None
    raise ValueError(f"Unsupported LLM backend: {config.backend}")
