# -*- coding: utf-8 -*-
"""按类分层切验证集——只生成清单文件，不移动、不复制任何图片。

为什么分层：每个品种都固定抽 15% 出去，避免某个类比另一个类多抽一倍。
为什么不搬目录：数据是不可再生资产，搬坏了没法恢复；用清单切分，
              想退回只要把 manifests 目录删掉，原数据一个字节没动。

用法：
    python split_val.py                     # 两个数据集都切，15%
    python split_val.py --dataset pet --ratio 0.2 --seed 2026
"""
import argparse
import os
import random
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "manifests")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")


def collect(split_dir):
    """返回 {类别目录名: [图片绝对路径, ...]}，跳过空目录与非图片文件。"""
    by_cls = defaultdict(list)
    if not os.path.isdir(split_dir):
        return by_cls
    for cls in sorted(os.listdir(split_dir)):
        d = os.path.join(split_dir, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                by_cls[cls].append(os.path.join(d, f))
    return by_cls


def stratified(by_cls, ratio, seed):
    rng = random.Random(seed)
    train, val, report = [], [], []
    for cls, files in by_cls.items():
        n = len(files)
        if n < 2:
            train.extend(files)
            report.append((cls, n, 0, n))
            continue
        # 每类至少留 1 张做验证，也至少留 1 张做训练
        k = max(1, min(n - 1, int(round(n * ratio))))
        idx = list(range(n))
        rng.shuffle(idx)
        v = sorted(idx[:k])
        t = sorted(idx[k:])
        val.extend(files[i] for i in v)
        train.extend(files[i] for i in t)
        report.append((cls, n, len(v), len(t)))
    return train, val, report


def write_list(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(lines)) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="both", choices=["pet", "dogs", "both"])
    ap.add_argument("--ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    names = ["pet", "dogs"] if args.dataset == "both" else [args.dataset]
    ok = True
    for name in names:
        src = os.path.join(DATA, name, "train")
        by_cls = collect(src)
        if not by_cls:
            print("!! 没找到 %s，路径：%s" % (name, src))
            ok = False
            continue
        train, val, report = stratified(by_cls, args.ratio, args.seed)
        write_list(os.path.join(OUT, "%s_train.txt" % name), train)
        write_list(os.path.join(OUT, "%s_val.txt" % name), val)

        n_cls = len(report)
        min_val = min(r[2] for r in report)
        max_val = max(r[2] for r in report)
        print("\n[%s]  类别数=%d  总图=%d  训练=%d  验证=%d（占 %.1f%%）"
              % (name, n_cls, len(train) + len(val), len(train), len(val),
                 100.0 * len(val) / (len(train) + len(val))))
        print("     每类验证张数：%d ~ %d 张（分层生效）" % (min_val, max_val))
        print("     清单：manifests/%s_train.txt, manifests/%s_val.txt" % (name, name))
        print("     抽样前三类：" + "  ".join("%s:%d/%d" % (r[0], r[3], r[2]) for r in report[:3]))

    print("\n原数据未做任何改动，只新增/覆盖了 manifests 目录下的清单文件。" if ok else "\n有数据集没找到，检查路径。")


if __name__ == "__main__":
    main()
