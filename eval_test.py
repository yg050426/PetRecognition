# -*- coding: utf-8 -*-
"""在 Oxford-IIIT Pet 官方测试集上评测已训好的权重。

为什么必须单独做这一步：训练时用的 551 张验证集是自己从 train 里切的，
用来调参选模型；而承诺指标写的是"官方测试集"，两者口径不同，不能混说。

用法：
    python eval_test.py --ckpt checkpoints/pet_resnet18_finetune_best.pth --arch resnet18
    python eval_test.py --ckpt checkpoints/pet_resnet18_frozen_best.pth --arch resnet18
"""
import argparse
import csv
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = os.path.dirname(os.path.abspath(__file__))
NORMALIZE = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class TestSet(Dataset):
    """直接读 data/pet/test/<类别>/ 图片，类别名按 checkpoint 里存的顺序映射。"""

    def __init__(self, root, classes, img_size=224):
        self.tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(**NORMALIZE),
        ])
        self.c2i = {c: i for i, c in enumerate(classes)}
        self.items = []
        for cls in sorted(os.listdir(root)):
            d = os.path.join(root, cls)
            if not os.path.isdir(d) or cls not in self.c2i:
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    self.items.append((os.path.join(d, f), self.c2i[cls]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, lab = self.items[i]
        try:
            return self.tf(Image.open(p).convert("RGB")), lab
        except Exception:
            return torch.zeros(3, 224, 224), -1


def build(arch, num_classes, ckpt_path):
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise SystemExit("不认识 --arch")
    sd = torch.load(ckpt_path, map_location="cpu")
    m.load_state_dict(sd["model"])
    return m, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--split", default="test", choices=["test", "val"],
                    help="test=官方测试集(承诺指标口径)  val=自切验证集")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(args.ckpt, map_location="cpu")
    classes = sd["classes"]
    root = os.path.join(ROOT, "data", "pet", args.split)
    if not os.path.isdir(root):
        raise SystemExit("找不到 " + root)

    ds = TestSet(root, classes)
    dl = DataLoader(ds, args.batch, False, num_workers=2)
    model, sd = build(args.arch, len(classes), args.ckpt)
    model.to(device).eval()

    print("权重 = %s   口径 = %s   图片 = %d 张   类别 = %d"
          % (os.path.basename(args.ckpt), "官方测试集" if args.split == "test" else "自切验证集",
             len(ds), len(classes)))
    if "val_top1" in sd:
        print("权重里记录的训练时验证集 Top-1 = %.3f（这是选模型用的，不是测试集成绩）" % sd["val_top1"])

    correct = 0
    top5 = 0
    total = 0
    per_cls = {}
    conf = [[0] * len(classes) for _ in range(len(classes))]
    with torch.no_grad():
        for x, y in dl:
            keep = y >= 0
            x, y = x[keep].to(device), y[keep].to(device)
            if x.numel() == 0:
                continue
            out = model(x)
            _, p5 = out.topk(5, 1)
            pred = p5[:, 0]
            correct += (pred == y).sum().item()
            top5 += (p5 == y[:, None]).any(1).sum().item()
            total += y.size(0)
            for t_, p_ in zip(y.tolist(), pred.tolist()):
                conf[t_][p_] += 1
                d = per_cls.setdefault(t_, [0, 0])
                d[1] += 1
                if p_ == t_:
                    d[0] += 1

    acc1 = correct / total
    acc5 = top5 / total
    print("\nTop-1 = %.4f (%d/%d)" % (acc1, correct, total))
    print("Top-5 = %.4f (%d/%d)" % (acc5, top5, total))

    # 宏平均精确率/召回率/F1
    precs, recs = [], []
    for c in range(len(classes)):
        tp = conf[c][c]
        col = sum(conf[r][c] for r in range(len(classes)))
        row = sum(conf[c])
        precs.append(tp / col if col else 0.0)
        recs.append(tp / row if row else 0.0)
    macro_p = sum(precs) / len(precs)
    macro_r = sum(recs) / len(recs)
    f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) else 0.0
    print("宏平均精确率 = %.4f  宏平均召回率 = %.4f  宏平均F1 = %.4f" % (macro_p, macro_r, f1))

    worst = sorted(per_cls.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1))[:5]
    print("\n识别最差的 5 个类别（对/总）：")
    for c, (ok, n) in worst:
        print("   %s: %d/%d = %.1f%%" % (classes[c], ok, n, 100.0 * ok / max(n, 1)))

    outdir = os.path.join(ROOT, "logs")
    os.makedirs(outdir, exist_ok=True)
    base = os.path.basename(args.ckpt).replace(".pth", "")
    with open(os.path.join(outdir, "eval_%s_%s.csv" % (base, args.split)), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["class", "correct", "total", "accuracy"])
        for c in range(len(classes)):
            ok, n = per_cls.get(c, [0, 0])
            wr.writerow([classes[c], ok, n, round(ok / n, 4) if n else ""])
        wr.writerow([])
        wr.writerow(["Top1", round(acc1, 4)])
        wr.writerow(["Top5", round(acc5, 4)])
        wr.writerow(["macro_precision", round(macro_p, 4)])
        wr.writerow(["macro_recall", round(macro_r, 4)])
        wr.writerow(["macro_F1", round(f1, 4)])
    with open(os.path.join(outdir, "confusion_%s_%s.csv" % (base, args.split)), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["\\ 真↓ 预测→"] + classes)
        for r in range(len(classes)):
            wr.writerow([classes[r]] + conf[r])
    print("\n明细已存：logs/eval_%s_%s.csv 与 logs/confusion_%s_%s.csv" % (base, args.split, base, args.split))


if __name__ == "__main__":
    main()
