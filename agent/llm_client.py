# -*- coding: utf-8 -*-
"""
禾医·大模型客户端（OpenAI-compatible，文本 + 视觉）
==============================================================================
已实测可用的服务（2026-10-16 验证）:
  A. AMD LLMGateway（推荐，开箱即用，免费额度）:
       HEYI_LLM_BASE_URL     = https://developer.amd.com.cn/radeon/api/v1
       HEYI_LLM_API_KEY      = rc-xxxxxxxx（AMD 开发者门户申请）
       HEYI_LLM_MODEL        = DeepSeek-V4-Flash        # 文本，1M 上下文
       HEYI_LLM_VISION_MODEL = DeepSeek-V4.1-Flash      # 视觉，叶片照片第二意见
     可用模型清单（GET /models）: DeepSeek-V4.1-Flash(视觉) / DeepSeek-V4-Flash /
       DeepSeek-V4-Flash-Vision-Exp / GLM-5.3-Flash / Qwen3.8-27B(视觉) /
       Qwen3.8-Flash-Next(视觉) / MiMo-V2.6-Flash(视觉) / MiniCPM 系列(视觉)
  B. 华为云 ModelArts Studio(MaaS) 免费额度:
       HEYI_LLM_BASE_URL = https://api.modelarts-maas.com/v1
       HEYI_LLM_API_KEY  = <MaaS 控制台创建的 API Key>
       HEYI_LLM_MODEL    = DeepSeek-V3 / Qwen2.5-32B 等已部署模型名
  C. 任意 OpenAI 兼容服务（DeepSeek/Qwen/Ollama 本地）同理，仅改上述变量。

设计红线：LLM 只做"语言组织"与"症状描述"，**不做诊断事实生成**——
诊断结论全部来自 CNN 模型与知识库，规避大模型幻觉（防幻觉是答辩核心叙事）。
未配置或调用失败时自动降级为"知识库模板应答"，保证链路永不中断。
"""
import base64, json, os
import urllib.request


class LLMClient:
    def __init__(self):
        self.base = os.environ.get("HEYI_LLM_BASE_URL", "").rstrip("/")
        self.key = os.environ.get("HEYI_LLM_API_KEY", "")
        self.model = os.environ.get("HEYI_LLM_MODEL", "DeepSeek-V3")
        self.vision_model = os.environ.get("HEYI_LLM_VISION_MODEL", "DeepSeek-V4.1-Flash")
        self.enabled = bool(self.base and self.key)

    def chat(self, messages: list, temperature=0.4, timeout=45,
             model: str | None = None) -> str | None:
        if not self.enabled:
            return None
        payload = json.dumps({"model": model or self.model, "messages": messages,
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

    def chat_vision(self, prompt: str, image_path: str | None = None,
                    image_b64: str | None = None, temperature=0.3,
                    timeout=90) -> str | None:
        """视觉问答：叶片照片 + 文本提示 -> 症状描述文本。

        用途：CNN 低置信度样本的 VLM 第二意见（只描述可见症状、不给确诊），
        与 CNN 结论互相印证，进一步压低"把病叶判成健康"的风险。
        传入 image_path 或 image_b64 之一。
        """
        if not self.enabled:
            return None
        if image_b64 is None:
            if not image_path:
                return None
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]
        return self.chat([{"role": "user", "content": content}],
                         temperature=temperature, timeout=timeout,
                         model=self.vision_model)


if __name__ == "__main__":
    c = LLMClient()
    print("enabled:", c.enabled, "| text model:", c.model,
          "| vision model:", c.vision_model)
    if c.enabled:
        print(c.chat([{"role": "user", "content": "用一句话介绍番茄晚疫病"}]))
        v = c.chat_vision(
            "这是农作物叶片照片。只描述可见症状（病斑形态/颜色/位置），不要下确定结论。",
            image_path="data/demo_images/Tomato___Late_blight/0003faa8-4b27-4c65-bf42-6d9e352ca1a5___RS_Late.B 4946.JPG")
        print("vision:", (v or "")[:200])
