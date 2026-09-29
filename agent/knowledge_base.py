# -*- coding: utf-8 -*-
"""禾医·病虫害知识库检索（关键词+别名+作物过滤的轻量RAG）。"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(HERE, "..", "data", "knowledge", "diseases.json")

class KnowledgeBase:
    def __init__(self, path=KB_PATH):
        raw = json.load(open(path, encoding="utf-8"))
        self.diseases = raw["diseases"]
        self.quick = self.diseases.get("_其余32类速查", {})
        self.index = {}
        for class_id, info in {**{k: v for k, v in self.diseases.items() if not k.startswith("_")},
                               **{k: v for k, v in self.quick.items() if not k.startswith("_")}}.items():
            tokens = set(re.findall(r"[\u4e00-\u9fff]+|[A-Za-z_]{3,}", class_id + "|" +
                                    json.dumps(info, ensure_ascii=False)))
            self.index[class_id] = tokens

    def get(self, class_name: str) -> dict:
        """按PlantVillage类别名取完整方案。"""
        if class_name in self.diseases and not class_name.startswith("_"):
            return self.diseases[class_name]
        base = class_name
        if base in self.quick:
            full = dict(self.diseases.get(base, {}))
            full.update(self.quick[base])
            return full
        return {"crop": "?", "name_cn": class_name, "pathogen": "未知",
                "symptoms": "知识库暂无记录", "prevention": "-", "treatment": "-"}

    def search(self, query: str, topk: int = 3) -> list:
        """中文/英文关键词检索, 返回[(class_id, score, info)]。"""
        q = set(re.findall(r"[\u4e00-\u9fff]+|[A-Za-z_]{3,}", query.lower()))
        scored = []
        for cid, tokens in self.index.items():
            s = 0.0
            for t in q:
                for tok in tokens:
                    tl = tok.lower()
                    if t == tl:
                        s += 2.0
                    elif len(t) >= 2 and (t in tl or tl in t):
                        s += 0.8
            if s > 0:
                scored.append((cid, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(cid, s, self.get(cid)) for cid, s in scored[:topk]]

if __name__ == "__main__":
    kb = KnowledgeBase()
    print(kb.get("Tomato___Late_blight")["name_cn"])
    for cid, s, info in kb.search("番茄叶子长黑斑怎么治"):
        print(round(s, 2), cid, info["name_cn"])
