import sys, json, torch
sys.path.insert(0, "model")
from eval_v3 import CLASSES27

ckpts = sys.argv[1:-1] if len(sys.argv) > 2 else []
out = sys.argv[-1]
assert len(ckpts) >= 2, "usage: soup.py ckpt1 ckpt2 ... out.pth"

sds = []
meta = None
for p in ckpts:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd = {k[len("backbone."):]: v for k, v in ck["model"].items() if k.startswith("backbone.")}
    sds.append(sd)
    meta = {"tags": meta.get("tags", []) + [ck.get("tag", p)], "classes": ck["classes"]} if meta else \
           {"tags": [ck.get("tag", p)], "classes": ck["classes"]}
print("参与模型:", meta["tags"])
avg = {}
for k in sds[0]:
    acc = sds[0][k].float()
    for sd in sds[1:]:
        acc = acc + sd[k].float()
    avg[k] = (acc / len(sds)).to(sds[0][k].dtype)
torch.save({"model": avg, "classes": meta["classes"], "soup": meta["tags"]}, out)
print("soup 已保存:", out)
