# -*- coding: utf-8 -*-
"""宠物品种识别训练脚本（迁移学习，两阶段：先冻结骨干，再解冻微调）。

用法：
    python train.py --smoke                      # 只跑 2 个 batch，验证通路，几十秒
    python train.py --dataset pet --arch resnet18 --stage frozen --epochs 10
    python train.py --dataset pet --arch resnet18 --stage finetune --epochs 30 --lr 1e-4

看到什么算成功：
    --smoke  能打印出 loss 和 Top-1，说明 数据→模型→反向→优化器 整条链路通了
    frozen   第一轮 val Top-1 明显高于瞎猜（pet 37 类瞎猜 = 2.7%）
    finetune val Top-1 比 frozen 阶段再往上走

会怎么翻车（按发生率排序）：
    1. 忘了 model.eval()  → 验证时 Dropout/BN 还在闹，分数莫名偏低
    2. 忘了 zero_grad()   → loss 曲线乱跳
    3. 类别顺序不一致      → 训练正常、指标漂亮，但类别全串行（不报错的错）
       对策：清单文件里带完整路径，类别名取目录名，两边同一套排序
    4. 显存爆             → 8GB 上 batch 32 + 224 是安全的；爆了就 --batch 16
"""
import argparse
import csv
import os
import random
import sys
import time

# Windows 控制台默认 GBK，打印 ✓ ↑ 这类字符会直接崩；统一改 UTF-8 并容错
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
MAN = os.path.join(ROOT, "manifests")
LOGDIR = os.path.join(ROOT, "logs")
CKPT = os.path.join(ROOT, "checkpoints")

NORMALIZE = dict(
    mean=[0.485, 0.456, 0.406],      # ImageNet 统计量，用预训练权重就必须用这套归一化
    std=[0.229, 0.224, 0.225],
)


class ListDataset(Dataset):
    """从清单文件读图片路径，类别名 = 图片所在目录名。"""

    def __init__(self, list_path, img_size, train):
        with open(list_path, encoding="utf-8") as fh:
            self.paths = [l.strip() for l in fh if l.strip()]
        self.classes = sorted({os.path.basename(os.path.dirname(p)) for p in self.paths})
        self.c2i = {c: i for i, c in enumerate(self.classes)}
        if train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(**NORMALIZE),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(**NORMALIZE),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        try:
            img = Image.open(p).convert("RGB")
            img = self.tf(img)
        except Exception:
            # 坏图不能让整个训练崩，返回一张黑图并标 -1，训练循环里跳过
            img = torch.zeros(3, 224, 224)
            return img, -1
        return img, self.c2i[os.path.basename(os.path.dirname(p))]


def build_model(arch, num_classes, stage, device):
    weights = "IMAGENET1K_V1" if arch in ("resnet18", "resnet50") else "DEFAULT"
    if arch == "resnet18":
        m = models.resnet18(weights=weights)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        trainable = m.fc.parameters() if stage == "frozen" else m.parameters()
    elif arch == "resnet50":
        m = models.resnet50(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        trainable = m.fc.parameters() if stage == "frozen" else m.parameters()
    elif arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        trainable = m.classifier.parameters() if stage == "frozen" else m.parameters()
    else:
        raise SystemExit("不认识的 --arch：%s" % arch)

    if stage == "frozen":
        for p in m.parameters():
            p.requires_grad = False
        # 最后那层是新初始化出来的，必须重新打开放 trainable
        last = m.fc if arch != "mobilenet_v3_small" else m.classifier[-1]
        for p in last.parameters():
            p.requires_grad = True
    m.to(device)
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in m.parameters())
    print("骨干=%s  阶段=%s  可训练参数 %.2fM / 总 %.2fM"
          % (arch, stage, n_train / 1e6, n_all / 1e6))
    return m, [p for p in m.parameters() if p.requires_grad]


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()                       # 关键：验证必须切 eval
    loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        if int(y.min()) < 0:
            keep = y >= 0
            x, y = x[keep], y[keep]
            if x.numel() == 0:
                continue
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss += criterion(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    model.train()                      # 评估完切回训练
    return loss / max(total, 1), correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pet", choices=["pet", "dogs"])
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--stage", default="frozen", choices=["frozen", "finetune"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true", help="只跑 2 个 batch 验证通路")
    args = ap.parse_args()

    torch.manual_seed(2026)
    random.seed(2026)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("设备 =", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

    tr_list = os.path.join(MAN, "%s_train.txt" % args.dataset)
    va_list = os.path.join(MAN, "%s_val.txt" % args.dataset)
    if not os.path.exists(tr_list):
        raise SystemExit("找不到清单 %s\n先跑：python split_val.py --dataset %s" % (tr_list, args.dataset))

    ds_tr = ListDataset(tr_list, args.img, True)
    ds_va = ListDataset(va_list, args.img, False)
    print("训练 %d 张 / 验证 %d 张 / 类别 %d 个" % (len(ds_tr), len(ds_va), len(ds_tr.classes)))

    dl_tr = DataLoader(ds_tr, args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, args.batch, False, num_workers=args.workers, pin_memory=True)

    model, trainable = build_model(args.arch, len(ds_tr.classes), args.stage, device)
    lr = args.lr if args.lr else (1e-3 if args.stage == "frozen" else 1e-4)
    opt = torch.optim.Adam(trainable, lr=lr)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    os.makedirs(CKPT, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)
    log_path = os.path.join(LOGDIR, "train_%s_%s_%s.csv" % (args.dataset, args.arch, args.stage))
    fh = open(log_path, "w", newline="", encoding="utf-8")
    wr = csv.writer(fh)
    wr.writerow(["epoch", "train_loss", "val_loss", "val_top1", "lr", "sec", "gpu_gb"])

    best = -1.0
    best_path = os.path.join(CKPT, "%s_%s_%s_best.pth" % (args.dataset, args.arch, args.stage))
    epochs = 1 if args.smoke else args.epochs

    for ep in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        n_batch = 0
        nb_correct = 0
        nb_total = 0
        t0 = time.time()
        for x, y in dl_tr:
            if int(y.min()) < 0:
                keep = y >= 0
                x, y = x[keep], y[keep]
                if x.numel() == 0:
                    continue
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()                       # ③ 不清零就会累积上一批的梯度
            out = model(x)                        # ④ 前向
            loss = crit(out, y)                   # ⑤ 算错多少
            loss.backward()                       # ⑥ 反向求梯度
            opt.step()                            # ⑦ 调参数
            run_loss += loss.item()
            n_batch += 1
            nb_correct += (out.argmax(1) == y).sum().item()
            nb_total += y.size(0)
            if args.smoke and n_batch >= 2:
                break

        vl, vt1 = evaluate(model, dl_va, device, crit) if not args.smoke else (float("nan"), float("nan"))
        sched.step()
        gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if device == "cuda" else 0.0
        el = time.time() - t0
        print("ep%02d  train_loss=%.4f  train_top1=%.3f  val_loss=%.4f  val_top1=%.3f  %.0fs  显存峰值%.2fGB"
              % (ep, run_loss / max(n_batch, 1), nb_correct / max(nb_total, 1), vl, vt1, el, gb))
        wr.writerow([ep, round(run_loss / max(n_batch, 1), 5), round(vl, 5), round(vt1, 5),
                     opt.param_groups[0]["lr"], round(el, 1), round(gb, 2)])
        fh.flush()

        if not args.smoke and vt1 > best:
            best = vt1
            torch.save({"model": model.state_dict(), "classes": ds_tr.classes,
                        "arch": args.arch, "val_top1": vt1}, best_path)
            print("   * 新最好 val Top-1 = %.3f，已存 %s" % (vt1, os.path.basename(best_path)))

    if args.smoke:
        print("\n[OK] 通路验证通过：数据读取 → 前向 → 损失 → 反向 → 更新 全跑通了。")
        print("  下一步：python train.py --dataset %s --arch %s --stage frozen --epochs 10"
              % (args.dataset, args.arch))
    else:
        fh.close()
        print("\n最好 val Top-1 = %.3f，权重在 %s，日志在 %s" % (best, best_path, log_path))
        print("[留图] 这一轮的数据先变成论文里的图，再干别的：")
        print("       cd /d %s && E:\\Miniconda3\\envs\\pet\\python.exe make_figs.py" % ROOT)


if __name__ == "__main__":
    main()
