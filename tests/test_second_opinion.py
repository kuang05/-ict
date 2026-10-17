# -*- coding: utf-8 -*-
"""VLM 第二意见仲裁 · 单元测试（不依赖真实 API）
验证三态逻辑：agree / disagree / uncertain + 降级 + 触发条件。
运行: python -m pytest tests/ -q  （在 he-yi 仓库根目录）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import second_opinion as so

CANDS = [{"class_name": "Tomato___Late_blight", "name_cn": "番茄晚疫病"},
         {"class_name": "Tomato___Early_blight", "name_cn": "番茄早疫病"},
         {"class_name": "Tomato___healthy", "name_cn": "番茄健康叶片"}]
NAMES = [c["name_cn"] for c in CANDS]


class FakeLLM:
    def __init__(self, reply, enabled=True):
        self.reply = reply
        self.enabled = enabled

    def chat_vision(self, prompt, image_path=None, image_b64=None, temperature=0.1, timeout=60):
        return self.reply


def test_should_review_threshold():
    assert so.should_review(0.55, tau=0.6) is True
    assert so.should_review(0.87, tau=0.6) is False
    assert so.should_review(None) is False


def test_agree_when_vlm_picks_top1():
    r = so.arbitrate(cnn_top3=CANDS, names_cn=NAMES, llm=FakeLLM("看到水浸状暗斑。\n结论: 1"))
    assert r["called"] and r["verdict"] == "agree" and r["vlm_choice"] == "番茄晚疫病"
    assert "双通道一致" in so.arbitrate_note(r)


def test_disagree_when_vlm_picks_other():
    r = so.arbitrate(cnn_top3=CANDS, names_cn=NAMES, llm=FakeLLM("病斑有同心轮纹。\n结论: 2"))
    assert r["verdict"] == "disagree" and r["vlm_choice"] == "番茄早疫病"
    note = so.arbitrate_note(r)
    assert "人工复核" in note and "暂不按上方处方施药" in note


def test_uncertain_when_vlm_undecided():
    r = so.arbitrate(cnn_top3=CANDS, names_cn=NAMES, llm=FakeLLM("照片太模糊。\n结论: 0"))
    assert r["verdict"] == "uncertain"
    assert "无法确定" in so.arbitrate_note(r)


def test_degrade_without_llm():
    r = so.arbitrate(cnn_top3=CANDS, names_cn=NAMES, llm=FakeLLM("x", enabled=False))
    assert r["called"] is False and r["verdict"] == "uncertain"


def test_degrade_on_empty_reply():
    r = so.arbitrate(cnn_top3=CANDS, names_cn=NAMES, llm=FakeLLM(None))
    assert r["called"] is True and r["verdict"] == "uncertain"


def test_no_candidates():
    r = so.arbitrate(cnn_top3=[], names_cn=[], llm=FakeLLM("结论: 1"))
    assert r["called"] is False
