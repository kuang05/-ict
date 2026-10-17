# -*- coding: utf-8 -*-
"""OpenAI兼容LLM客户端(华为云MaaS / AMD LLMGateway 等)。未配置Key时 llm.enabled=False, 全链路降级为知识库模板模式。"""
import json, os, urllib.request

class LLMClient:
    def __init__(self):
        self.base = os.environ.get("HEYI_LLM_BASE_URL", "").rstrip("/")
        self.key = os.environ.get("HEYI_LLM_API_KEY", "")
        self.model = os.environ.get("HEYI_LLM_MODEL", "DeepSeek-V3")
        self.enabled = bool(self.base and self.key)

    @staticmethod
    def _extract_content(msg):
        """兼容三种返回: 标准 content / 思考型(Qwen3.8: reasoning + 空 content) / delta。"""
        if not isinstance(msg, dict):
            return None
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content
        reasoning = msg.get("reasoning") or msg.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            # 思考型模型: content 为空时从 reasoning 提取最终结论(取最后一段非空行)
            lines = [l.strip() for l in reasoning.splitlines() if l.strip()]
            return lines[-1] if lines else None
        return None

    def chat(self, messages, temperature=0.3, timeout=45):
        if not self.enabled:
            return None
        payload = json.dumps({"model": self.model, "messages": messages,
                              "temperature": temperature}).encode()
        req = urllib.request.Request(f"{self.base}/chat/completions", data=payload, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.key}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            return self._extract_content(data["choices"][0]["message"])
        except Exception as e:
            print("[HeYi-LLM] fallback:", e)
            return None
