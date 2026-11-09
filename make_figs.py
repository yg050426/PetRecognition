# -*- coding: utf-8 -*-
"""把毕设每个阶段的数据变成论文里能直接插的图。

为什么要单独有这个脚本：训练和评测只留下 CSV 和文字日志，写论文时要一张张手画，
既容易和真实数据不一致，也来不及。这个脚本规定了一整套图的编号和内容，
每次跑完实验敲一句，图就全部重画一遍，数据和图永远对得上。

用法：
    python make_figs.py                 # 画全部（有数据就画，没数据跳过并说明）
    python make_figs.py --only 5-03     # 只画某一张
    python make_figs.py --ckpt checkpoints/pet_resnet50_finetune_best.pth --arch resnet50
                                        # 顺带重画预测样例图（要用 GPU，跑推理）

产物：
    figures/图5-XX_中文名.png           300 dpi，宽 15 cm，直接进论文
    figures/图注清单.md                 图号 ↔ 论文放哪 ↔ 数据来自哪个文件
    figures/生成记录.json               每张图的生成时间、来源文件、来源文件指纹
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "logs")
FIGS = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data", "pet")

CM = 1 / 2.54
FIG_W = 15 * CM          # A4 去掉 3cm 页边距后的正文宽度
DPI = 300

# ---------------------------------------------------------------- 中文字体
def setup_font():
    for name in ("simhei.ttf", "msyh.ttc", "simsun.ttc"):
        p = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", name)
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
    prefer = []
    for cand in ("SimHei", "Microsoft YaHei", "SimSun"):
        for f in font_manager.fontManager.ttflist:
            if f.name == cand and cand not in prefer:
                prefer.append(cand)
    plt.rcParams["font.sans-serif"] = prefer + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = DPI
    plt.rcParams["font.size"] = 8.5
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.7


# ---------------------------------------------------------------- 小工具
def wrap_name(name, width=15):
    """品种名按空格折行，超过一行再截断，避免横向压到隔壁格子。"""
    import textwrap
    lines = textwrap.wrap(name, width) or [name]
    if len(lines) > 2:
        lines = lines[:2]
        lines[1] = lines[1][:width - 1] + "…"
    return "\n".join(lines)


def clip(name, width):
    """硬截断成一行，保证不越界。"""
    return name if len(name) <= width else name[:width - 1] + "…"


def oneline(name, width=22):
    """预测错时要把预测名塞进一行，太长就缩。"""
    abbr = {"terrier": "terr.", "shorthaired": "shorthd.", "spaniel": "sp.",
            "setter": "set.", "hound": "hnd.", "retriever": "retr."}
    parts = name.split()
    parts = [abbr.get(p, p) for p in parts]
    out = " ".join(parts)
    return out if len(out) <= width else out[:width - 1] + "…"


def square(im, side=224):
    """居中裁成正方形再缩放。不裁的话每张图高矮不一，格子标题会互相压住。"""
    w, h = im.size
    m = min(w, h)
    im = im.crop(((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2))
    return im.resize((side, side))


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.reader(f))


def load_train(tag):
    rows = read_csv(os.path.join(LOGS, "train_%s.csv" % tag))
    h = rows[0]
    out = {"epoch": [], "train_loss": [], "val_loss": [], "val_top1": [], "lr": [], "sec": []}
    for r in rows[1:]:
        if len(r) < len(h):
            continue
        for k, v in zip(h, r):
            if k in out:
                out[k].append(float(v))
    return out


def class_names():
    p = os.path.join(DATA, "class_names.csv")
    if not os.path.exists(p):
        return ["%02d" % i for i in range(37)]
    rows = read_csv(p)
    names = []
    for r in rows[1:]:
        if len(r) >= 2:
            n = r[1].replace("_", " ")
            names.append(n[0].upper() + n[1:])
    return names


def tag_cn(tag):
    """pet_resnet18_frozen -> ResNet18 冻结骨干"""
    arch = "ResNet50" if "resnet50" in tag else ("ResNet18" if "resnet18" in tag else "其他")
    stage = {"frozen": "冻结骨干", "finetune": "全网络微调"}.get(tag.split("_")[-1], tag.split("_")[-1])
    return "%s %s" % (arch, stage)


def save(fig, path, caption, source):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    st = os.stat(source[0]) if source and os.path.exists(source[0]) else None
    MADE.append({
        "file": os.path.basename(path),
        "caption": caption,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": [os.path.relpath(s, ROOT).replace("\\", "/") for s in source] if source else [],
        "source_mtime": [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.stat(s).st_mtime))
                         for s in source if os.path.exists(s)] if source else [],
    })
    print("  [OK] %-52s  <- %s" % (os.path.basename(path),
                                   ", ".join(os.path.basename(s) for s in source) if source else "-"))


def sha(p):
    if not os.path.exists(p):
        return None
    h = hashlib.sha1()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:12]


MADE = []
SKIPPED = []

# ---------------------------------------------------------------- 图 5-01 数据集样本
def fig_5_01():
    src = [os.path.join(DATA, "test")]
    names = class_names()
    cells, labels = [], []
    for i in range(len(names)):
        d = os.path.join(DATA, "test", "%02d" % i)
        if not os.path.isdir(d):
            continue
        fs = sorted(glob.glob(os.path.join(d, "*.jpg")))
        if not fs:
            continue
        cells.append(fs[0]); labels.append("%02d\n%s" % (i, wrap_name(names[i])))
    if not cells:
        SKIPPED.append(("图5-01", "data/pet/test 不存在"))
        return
    from PIL import Image
    n = len(cells)
    ncol = 6
    nrow = int(np.ceil(n / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(FIG_W, nrow * 2.35 * CM))
    ax = np.array(ax).reshape(-1)
    for k in range(len(ax)):
        ax[k].axis("off")
        if k >= n:
            continue
        try:
            im = square(Image.open(cells[k]).convert("RGB"), 200)
        except Exception:
            continue
        ax[k].imshow(im)
        ax[k].set_title(labels[k], fontsize=5.6, pad=2.2)
    fig.subplots_adjust(hspace=0.42, wspace=0.06, top=0.94)
    fig.suptitle("Oxford-IIIT Pet 数据集 37 个类别测试集样本示例（每类取 1 张）", fontsize=9, y=0.995)
    save(fig, os.path.join(FIGS, "图5-01_数据集类别样本示例.png"),
         "图5-1 Oxford-IIIT Pet 数据集 37 个类别测试集样本示例", ["data/pet/class_names.csv"] + src)


# ---------------------------------------------------------------- 图 5-02 样本数量分布
def fig_5_02():
    names = class_names()
    tr = [0] * 37
    va = [0] * 37
    te = [0] * 37
    for line in open(os.path.join(ROOT, "manifests", "pet_train.txt"), encoding="utf-8"):
        i = int(os.path.basename(line.strip().split("\\")[-2])) if line.strip() else None
        if i is not None:
            tr[i] += 1
    for line in open(os.path.join(ROOT, "manifests", "pet_val.txt"), encoding="utf-8"):
        i = int(os.path.basename(line.strip().split("\\")[-2]))
        va[i] += 1
    for i in range(37):
        d = os.path.join(DATA, "test", "%02d" % i)
        te[i] = len(glob.glob(os.path.join(d, "*.jpg"))) if os.path.isdir(d) else 0
    src = ["manifests/pet_train.txt", "manifests/pet_val.txt", "data/pet/test"]
    x = np.arange(37)
    fig, ax = plt.subplots(figsize=(FIG_W, 5.2 * CM))
    ax.bar(x - 0.28, tr, 0.28, label="训练集（切分后 %d 张）" % sum(tr), color="#2f6f9f")
    ax.bar(x, va, 0.28, label="验证集（内部切出 %d 张）" % sum(va), color="#6aab8e")
    ax.bar(x + 0.28, te, 0.28, label="官方测试集（%d 张）" % sum(te), color="#c98a3b")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=5.2)
    ax.set_ylabel("图片数量（张）")
    ax.set_ylim(0, max(tr) * 1.16)
    ax.legend(fontsize=7.2, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.60), frameon=False)
    ax.grid(axis="y", lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    fig.suptitle("各品种样本数量分布", fontsize=9, y=1.005)
    save(fig, os.path.join(FIGS, "图5-02_各类别样本数量分布.png"),
         "图5-2 各品种在训练/验证/测试三个子集上的样本数量分布", src)


# ---------------------------------------------------------------- 图 5-03..05 单组训练曲线
def fig_curve(tag, no):
    p = os.path.join(LOGS, "train_%s.csv" % tag)
    if not os.path.exists(p):
        SKIPPED.append(("图%s" % no, "缺 logs/train_%s.csv" % tag))
        return
    d = load_train(tag)
    src = ["logs/train_%s.csv" % tag]
    ep = d["epoch"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIG_W, 5.4 * CM))
    a1.plot(ep, d["train_loss"], "o-", ms=3.2, lw=1.1, color="#2f6f9f", label="训练集损失")
    a1.plot(ep, d["val_loss"], "s--", ms=3.2, lw=1.1, color="#c0504d", label="验证集损失")
    a1.set_xlabel("轮次 (epoch)"); a1.set_ylabel("交叉熵损失"); a1.legend(fontsize=7.5)
    a1.set_title("(a) 损失曲线", fontsize=8.5)
    a1.grid(lw=0.4, alpha=0.35); a1.set_axisbelow(True)

    a2.plot(ep, [v * 100 for v in d["val_top1"]], "o-", ms=3.2, lw=1.2, color="#3a7d44", label="验证集 Top-1")
    bi = int(np.argmax(d["val_top1"]))
    a2.plot(ep[bi], d["val_top1"][bi] * 100, "*", ms=11, color="#d1495b",
            label="最好一轮 ep%02d %.2f%%" % (ep[bi], d["val_top1"][bi] * 100))
    a2.axhline(90, color="#888888", lw=0.8, ls=":", label="开题承诺线 90%（验证集口径）")
    a2.set_xlabel("轮次 (epoch)"); a2.set_ylabel("Top-1 准确率 (%)")
    a2.set_ylim(min(75, min(d["val_top1"]) * 100 - 2), max(96, max(d["val_top1"]) * 100 + 2))
    a2.legend(fontsize=6.8, loc="lower right", framealpha=0.92)
    a2.set_title("(b) 验证集 Top-1 准确率", fontsize=8.5)
    a2.grid(lw=0.4, alpha=0.35); a2.set_axisbelow(True)
    fig.suptitle("%s 训练过程曲线（%d 轮）" % (tag_cn(tag), len(ep)), fontsize=9, y=1.07)
    save(fig, os.path.join(FIGS, "图%s_训练曲线_%s.png" % (no, tag)),
         "图%s %s 训练过程中的损失与验证集 Top-1 曲线" % (disp(no), tag_cn(tag)), src)


# ---------------------------------------------------------------- 图 5-06 多组对比
def fig_5_06(tags):
    have = [t for t in tags if os.path.exists(os.path.join(LOGS, "train_%s.csv" % t))]
    if len(have) < 2:
        SKIPPED.append(("图5-06", "可用训练日志不足 2 组"))
        return
    colors = ["#2f6f9f", "#c0504d", "#3a7d44", "#8064a2", "#d18b2b"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIG_W, 5.6 * CM))
    src = []
    for i, t in enumerate(have):
        d = load_train(t)
        src.append("logs/train_%s.csv" % t)
        a1.plot(d["epoch"], d["train_loss"], "o-", ms=2.6, lw=1.0, color=colors[i], label=tag_cn(t))
        a2.plot(d["epoch"], [v * 100 for v in d["val_top1"]], "o-", ms=2.6, lw=1.2, color=colors[i], label=tag_cn(t))
    a2.axhline(90, color="#888888", lw=0.9, ls=":")
    a1.set_xlabel("轮次 (epoch)"); a1.set_ylabel("训练集损失"); a1.set_title("(a) 训练损失对比", fontsize=8.5)
    a2.set_xlabel("轮次 (epoch)"); a2.set_ylabel("验证集 Top-1 (%)"); a2.set_title("(b) 验证集 Top-1 对比", fontsize=8.5)
    a2.text(0.6, 90.6, "90%（验证集口径）", fontsize=6.8, color="#666666",
            bbox=dict(fc="white", ec="none", pad=1.0))
    for a in (a1, a2):
        a.grid(lw=0.4, alpha=0.35); a.set_axisbelow(True)
        a.legend(fontsize=7.2, loc="lower right" if a is a2 else "upper right")
    fig.suptitle("不同骨干网络 / 训练策略的收敛过程对比", fontsize=9, y=1.07)
    save(fig, os.path.join(FIGS, "图5-06_多组配置训练曲线对比.png"),
         "图5-6 三组实验配置的损失与验证集 Top-1 收敛曲线对比", src)


# ---------------------------------------------------------------- 图 5-07/08 混淆矩阵
def fig_conf(tag, no):
    p = os.path.join(LOGS, "confusion_%s_best_test.csv" % tag)
    if not os.path.exists(p):
        SKIPPED.append(("图%s" % no, "缺 logs/confusion_%s_best_test.csv" % tag))
        return
    rows = read_csv(p)
    m = np.array([[float(v) for v in r[1:]] for r in rows[1:]], dtype=float)
    names = class_names()
    fig, ax = plt.subplots(figsize=(FIG_W, 7.4 * CM))
    im = ax.imshow(m, cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(37)); ax.set_yticks(range(37))
    ax.set_xticklabels(names, rotation=90, fontsize=4.6)
    ax.set_yticklabels(names, fontsize=4.6)
    for i in range(37):
        for j in range(37):
            if m[i, j] >= 1:
                ax.text(j, i, "%d" % m[i, j], ha="center", va="center", fontsize=3.6,
                        color="white" if m[i, j] > m.max() * 0.55 else "#1a1a1a")
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("样本数（张）", fontsize=7.5); cb.ax.tick_params(labelsize=6.5)
    ax.set_title("%s 在官方测试集上的混淆矩阵（共 %d 张）" % (tag_cn(tag), int(m.sum())), fontsize=8.8, pad=6)
    save(fig, os.path.join(FIGS, "图%s_混淆矩阵_%s.png" % (no, tag)),
         "图%s %s 在官方测试集上的混淆矩阵" % (disp(no), tag_cn(tag)), ["logs/confusion_%s_best_test.csv" % tag])


# ---------------------------------------------------------------- 图 5-09 各类别准确率
def fig_5_09(tags, no="5-10"):
    have = [t for t in tags if os.path.exists(os.path.join(LOGS, "eval_%s_best_test.csv" % t))]
    if not have:
        SKIPPED.append(("图" + no, "没有 eval_*.csv"))
        return
    colors = ["#2f6f9f", "#c0504d", "#3a7d44", "#8064a2"]
    names = class_names()
    fig, ax = plt.subplots(figsize=(FIG_W, 6.2 * CM))
    w = 0.8 / len(have)
    src = []
    for i, t in enumerate(have):
        rows = read_csv(os.path.join(LOGS, "eval_%s_best_test.csv" % t))
        acc = [float(r[3]) * 100 for r in rows[1:] if len(r) >= 4 and r[0].strip().isdigit()]
        if len(acc) != 37:
            continue
        src.append("logs/eval_%s_best_test.csv" % t)
        ax.bar(np.arange(37) + i * w, acc, w, color=colors[i % 4], label="%s 均值 %.2f%%" % (tag_cn(t), np.mean(acc)))
    ax.axhline(90, color="#888888", lw=0.9, ls=":")
    ax.set_ylim(0, 112)
    ax.text(-0.30, 91.8, "90%（承诺线）", fontsize=6.2, color="#666666", ha="left")
    ax.set_xticks(np.arange(37) + w * (len(have) - 1) / 2)
    ax.set_xticklabels(names, rotation=90, fontsize=5)
    ax.set_ylabel("Top-1 准确率 (%)"); ax.set_xlabel("品种类别")
    ax.legend(fontsize=7, ncol=len(have), loc="upper center",
              bbox_to_anchor=(0.5, -0.60), frameon=False)
    ax.grid(axis="y", lw=0.4, alpha=0.35); ax.set_axisbelow(True)
    fig.suptitle("各品种 Top-1 准确率对比（官方测试集）", fontsize=9, y=1.0)
    save(fig, os.path.join(FIGS, "图%s_各类别准确率对比.png" % no),
         "图%s 各实验配置在 37 个品种上的 Top-1 准确率对比" % disp(no), src)


# ---------------------------------------------------------------- 图 5-10 实验结果汇总（表转图）
def fig_table1(tags, no="5-01"):
    rows = []
    src = []
    for t in tags:
        ev = os.path.join(LOGS, "eval_%s_best_test.csv" % t)
        trc = os.path.join(LOGS, "train_%s.csv" % t)
        if not (os.path.exists(ev) and os.path.exists(trc)):
            continue
        e = read_csv(ev)
        met = {r[0]: r[1] for r in e if len(r) >= 2 and not r[0].strip().isdigit()}
        d = load_train(t)
        bi = int(np.argmax(d["val_top1"]))
        rows.append([tag_cn(t), len(d["epoch"]), "ep%02d" % ep_txt(bi),
                     "%.2f%%" % (d["val_top1"][bi] * 100),
                     pct(met.get("Top1")), pct(met.get("Top5")),
                     pct(met.get("macro_F1")), "%.1f s" % np.mean(d["sec"])])
        src += [ev, trc]
    if not rows:
        SKIPPED.append(("表" + disp(no), "没有同时具备训练与测试日志的配置"))
        return
    head = ["实验配置", "轮数", "最好轮次\n(首次达到)", "验证集\nTop-1", "测试集\nTop-1", "测试集\nTop-5", "宏平均\nF1", "平均每轮\n耗时"]

    fig, ax = plt.subplots(figsize=(FIG_W, 1.15 * CM * (len(rows) + 1.4)))
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=head, cellLoc="center", colWidths=[0.28, 0.07, 0.09, 0.11, 0.11, 0.11, 0.10, 0.13], loc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(7.6); tb.scale(1, 1.55)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#9a9a9a"); cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#2f6f9f"); cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#eef3f8")
    ax.set_title("表%s 各实验配置在验证集与官方测试集上的指标汇总" % disp(no), fontsize=9, pad=8)
    save(fig, os.path.join(FIGS, "表%s_实验结果汇总.png" % no),
         "表%s 各实验配置在验证集与官方测试集上的指标汇总" % disp(no), src)


def ep_txt(i):
    return i + 1


def disp(no):
    a, b = no.split("-")
    return "%s-%d" % (a, int(b))


def pct(v):
    try:
        return "%.2f%%" % (float(v) * 100)
    except Exception:
        return "-"


# ---------------------------------------------------------------- 图 5-11 预测样例（需推理）
def fig_5_11(arch, ckpt, no="5-11"):
    if not ckpt or not os.path.exists(ckpt):
        SKIPPED.append(("图" + no, "未指定 --ckpt 或权重不存在"))
        return
    import torch
    from PIL import Image
    from torchvision import models, transforms
    sys.path.insert(0, ROOT)
    from eval_test import build, TestSet  # noqa
    names = class_names()
    classes = [n for n in names]
    model, sd = build(arch, len(classes), ckpt)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    rng = np.random.default_rng(20261109)
    picks, right, wrong = [], [], []
    for i in range(37):
        d = os.path.join(DATA, "test", "%02d" % i)
        fs = sorted(glob.glob(os.path.join(d, "*.jpg")))
        if not fs:
            continue
        f = fs[rng.integers(len(fs))]
        im = Image.open(f).convert("RGB")
        x = tf(im).unsqueeze(0).to(dev)
        with torch.no_grad():
            o = torch.softmax(model(x), 1)[0]
        pred, conf = int(o.argmax()), float(o.max())
        rec = (f, i, pred, conf, im)
        picks.append(rec)
        (right if pred == i else wrong).append(rec)
    ncol, nrow = 6, 6
    fig, axes = plt.subplots(nrow, ncol, figsize=(FIG_W, nrow * 2.75 * CM))
    axes = np.array(axes).reshape(-1)
    show = picks[:nrow * ncol]
    for k, ax in enumerate(axes):
        ax.axis("off")
        if k >= len(show):
            continue
        f, true, pred, conf, im = show[k]
        ax.imshow(square(im, 200))
        ok = (pred == true)
        t = "%s\n%s %d%%" % (clip(names[true], 15),
                               "√" if ok else "×" + clip(names[pred], 10),
                               int(round(conf * 100)))
        ax.set_title(t, fontsize=5.2, pad=2.4, color="#1a7f37" if ok else "#c0392b")
        if not ok:
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor="#c0392b", lw=1.6))
    fig.subplots_adjust(hspace=0.42, wspace=0.05, top=0.955)
    nr = sum(1 for s in show if s[1] == s[2])
    nw = len(show) - nr
    tagname = os.path.splitext(os.path.basename(ckpt))[0].replace("pet_", "").replace("_best", "")
    fig.suptitle("%s 测试集随机 %d 张预测结果（对 %d / 错 %d，这组样本正确率 %.1f%%）"
                 % (tag_cn(tagname), len(show), nr, nw, nr / max(len(show), 1) * 100), fontsize=9, y=0.998)
    save(fig, os.path.join(FIGS, "图%s_预测样例_真实与预测标签.png" % no),
         "图%s 测试集随机抽样预测结果（绿字预测正确，红框与红字预测错误；× 后为误判类别）" % disp(no), [ckpt])


# ---------------------------------------------------------------- 图注清单
def write_manifest():
    PAPER = [
        ("数据集类别样本示例", "第3章 3.2 数据集准备"),
        ("各类别样本数量分布", "第3章 3.2 数据集准备"),
        ("多组配置训练曲线对比", "第5章 5.4 对比实验分析"),
        ("训练曲线", "第5章 5.3 训练过程"),
        ("混淆矩阵", "第5章 5.5 错例分析"),
        ("各类别准确率对比", "第5章 5.4 对比实验分析"),
        ("实验结果汇总", "第5章 5.4 实验结果汇总（表）"),
        ("预测样例", "第5章 5.6 识别效果展示"),
    ]

    def paper_of(fn):
        for key, where in PAPER:
            if key in fn:
                return where
        return "—"
    lines = ["# 毕设实验图清单（自动生成的留图记录）\n",
             "> 生成时间：%s  ｜  命令：`python make_figs.py`  ｜  共 %d 张，跳过 %d 项\n"
             % (time.strftime("%Y-%m-%d %H:%M"), len(MADE), len(SKIPPED)),
             "| 文件名 | 图注（论文里用） | 论文位置 | 数据来源 | 来源文件时间 |",
             "| --- | --- | --- | --- | --- |"]
    for m in sorted(MADE, key=lambda x: x["file"]):
        lines.append("| `%s` | %s | %s | %s | %s |" % (
            m["file"], m["caption"], paper_of(m["file"]),
            "、".join(m["source"]) or "—",
            "、".join(m["source_mtime"]) or "—"))
    lines.append("")
    if SKIPPED:
        lines.append("## 本次没画出来的（原因如实记着，补齐数据后重跑本脚本即可）\n")
        for k, why in SKIPPED:
            lines.append("- **%s**：%s" % (k, why))
        lines.append("")
    with open(os.path.join(FIGS, "图注清单.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(FIGS, "生成记录.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "figures": MADE,
                   "skipped": [{"no": k, "why": w} for k, w in SKIPPED]}, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只画某图号，例如 5-03")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--arch", default="resnet18")
    args = ap.parse_args()

    setup_font()
    os.makedirs(FIGS, exist_ok=True)
    want = args.only

    # 顺序按论文里讲的先后：先冻结骨干基线，再解冻微调，再换更深的骨干
    PREFERRED = ["pet_resnet18_frozen", "pet_resnet18_finetune", "pet_resnet50_finetune"]
    found = sorted(os.path.basename(p)[len("train_"):-4]
                   for p in glob.glob(os.path.join(LOGS, "train_pet_*.csv")))
    tags = [t for t in PREFERRED if t in found] + [t for t in found if t not in PREFERRED]
    print("发现训练日志 %d 份：%s" % (len(tags), ", ".join(tags)))

    def run(no, fn):
        if want and want not in no:
            return
        try:
            fn()
        except Exception as e:
            print("  [FAIL] %s：%s: %s" % (no, type(e).__name__, e))
            SKIPPED.append(("图" + no, "%s: %s" % (type(e).__name__, e)))

    run("5-01", fig_5_01)
    run("5-02", fig_5_02)
    for i, t in enumerate(tags):
        run("5-%02d" % (3 + i), lambda t=t, i=i: fig_curve(t, "5-%02d" % (3 + i)))
    n = len(tags)
    if not want or want in "5-06":
        run("5-06", lambda: fig_5_06(tags))
    for i, t in enumerate(tags):
        run("5-%02d" % (7 + i), lambda t=t, i=i: fig_conf(t, "5-%02d" % (7 + i)))
    NO_PER, NO_SMP = "5-%02d" % (7 + n), "5-%02d" % (8 + n)
    run(NO_PER, lambda: fig_5_09(tags, NO_PER))
    run("表5-01", lambda: fig_table1(tags, "5-01"))
    if args.ckpt:
        run(NO_SMP, lambda: fig_5_11(args.arch, args.ckpt, NO_SMP))
    else:
        SKIPPED.append(("图" + NO_SMP, "加 --ckpt 和 --arch 才会画（要跑推理）"))

    write_manifest()
    print("\n完成：%d 张图，写在 %s" % (len(MADE), FIGS))
    if SKIPPED:
        print("跳过 %d 项：%s" % (len(SKIPPED), "; ".join("%s(%s)" % s for s in SKIPPED)))


if __name__ == "__main__":
    main()
