# -*- coding: utf-8 -*-
"""
禾医·大模型客户端（OpenAI-compatible，直连华为云 MaaS / 盘古 / 任意兼容服务）
==============================================================================
华为云接入（二选一，填入环境变量即可，无需改代码）:
  A. ModelArts Studio(MaaS) 免费额度:
       HEYI_LLM_BASE_URL = https://api.modelarts-maas.com/v1
       HEYI_LLM_API_KEY  = <MaaS 控制台创建的 API Key>
       HEYI_LLM_MODEL    = DeepSeek-V3 / Qwen2.5-32B 等已部署模型名
  B. 其他兼容服务(DeepSeek/Qwen/Ollama本地)同理, 仅改上述三个变量。
未配置或调用失败时自动降级为"知识库模板应答"，保证链路永不中断。
"""
import json, os, time
import urllib.request

class LLMClient:
    def __init__(self):
        self.base = os.environ.get("HEYI_LLM_BASE_URL", "").rstrip("/")
        self.key = os.environ.get("HEYI_LLM_API_KEY", "")
        self.model = os.environ.get("HEYI_LLM_MODEL", "DeepSeek-V3")
        self.enabled = bool(self.base and self.key)

    def chat(self, messages: list, temperature=0.4, timeout=45) -> str | None:
        if not self.enabled:
            return None
        payload = json.dumps({"model": self.model, "messages": messages,
                              "temperature": temperature}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}/chat/completions", data=payload, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print("[HeYi-LLM] call failed, fallback to KB template:", e)
            return None

if __name__ == "__main__":
    c = LLMClient()
    print("enabled:", c.enabled)
    if c.enabled:
        print(c.chat([{"role": "user", "content": "用一句话介绍番茄晚疫病"}]))
