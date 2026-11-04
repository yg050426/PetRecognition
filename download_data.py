# -*- coding: utf-8 -*-
"""下载两个官方数据集到 ./data
  1) Oxford-IIIT Pet —— 用 torchvision 内置下载器（图 + 标注，约 800MB）
  2) Stanford Dogs   —— torchvision 里没有这个数据集，必须从斯坦福官网下 tar 包（约 1GB）
可反复运行：已下好的部分会自动跳过。
"""
import os, sys, tarfile, urllib.request

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return "%.1f%s" % (n, u)
        n /= 1024.0
    return "%.1fTB" % n


def hook(block, bs, total):
    if total and total > 0 and block % 400 == 0:
        got = block * bs
        print("   ... %s / %s (%.1f%%)" % (human(got), human(total), got * 100.0 / total), flush=True)


# ---------- 1) Oxford-IIIT Pet ----------
try:
    from torchvision.datasets import OxfordIIITPet
    print("[1/2] Oxford-IIIT Pet 开始（species+breed，约 800MB）...", flush=True)
    OxfordIIITPet(root=DATA, split="train", target_types=("species", "breed"), download=True)
    OxfordIIITPet(root=DATA, split="test", target_types=("species", "breed"), download=True)
    print("[1/2] Oxford-IIIT Pet 完成", flush=True)
except Exception as e:
    print("[1/2] Pet 失败：%s: %s" % (type(e).__name__, e), flush=True)

# ---------- 2) Stanford Dogs ----------
print("[2/2] Stanford Dogs 开始（官方 tar，约 1GB）...", flush=True)
SD = os.path.join(DATA, "StanfordDogs")
os.makedirs(SD, exist_ok=True)
for fname, url in (
    ("images.tar", "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"),
    ("annotation.tar", "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"),
):
    dst = os.path.join(SD, fname)
    if os.path.exists(dst) and os.path.getsize(dst) > 10000:
        print("   已存在，跳过 %s" % fname, flush=True)
    else:
        try:
            urllib.request.urlretrieve(url, dst, hook)
            print("   %s 下载完成：%s" % (fname, human(os.path.getsize(dst))), flush=True)
        except Exception as e:
            print("   %s 下载失败：%s: %s" % (fname, type(e).__name__, e), flush=True)
            print("   手动下载备用地址：%s" % url, flush=True)
    if os.path.exists(dst) and os.path.getsize(dst) > 10000:
        try:
            with tarfile.open(dst) as t:
                t.extractall(SD)
            print("   %s 解压完成" % fname, flush=True)
        except Exception as e:
            print("   %s 解压失败：%s" % (fname, e), flush=True)

print("全部任务结束。", flush=True)
