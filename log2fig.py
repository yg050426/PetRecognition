# -*- coding: utf-8 -*-
"""把文字日志变成一张终端截图样式的图片，论文里可以直接当"实验过程证据"插。

为什么需要它：训练和评测的原始输出在 .log 里，老师看论文时想确认"这串数字是真跑出来的"，
一张带命令行、时间戳、文件指纹的终端图比正文更有说服力，而且比手动截屏整齐。

用法：
    python log2fig.py logs/run_frozen20.log
    python log2fig.py logs/run_frozen20.log logs/run_resnet50.log --out figures/证据
    python log2fig.py --tail 40 logs/eval_resnet50.log      # 只取最后 40 行
"""
import argparse
import hashlib
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
# 顺序：先挑带中文字形的，纯拉丁等宽字体（consola）放最后，只在全英文日志时才用得上
FONTS = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simsun.ttc",
         r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\consola.ttf",
         r"C:\Windows\Fonts\lucon.ttf", r"C:\Windows\Fonts\cour.ttf"]

WARN_BG = (72, 56, 20)
WARN_FG = (245, 198, 80)
BG = (24, 26, 31)
FG = (222, 226, 232)
DIM = (140, 148, 160)
ACC = (98, 196, 140)
TITLE_BG = (38, 41, 48)


def read_text_any(path):
    """按 UTF-8 -> GBK -> 兜底 的顺序试。返回 (文本, 实际编码)。
    这台机器上的日志两种编码都有，硬按一种读就会把中文读成乱码或方块。"""
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-8", "gbk", "cp936"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8(有坏字节)"


def has_cjk(font):
    """用"中"字试渲染，笔画像素太少就说明这个字体没有中文字形。"""
    try:
        m = font.getmask("中")
        return sum(1 for v in m if v) > 20
    except Exception:
        return False


def pick_font(px, need_cjk=True):
    """挑一个能同时显示中文和拉丁的字体；全英文日志才允许退回等宽 Consolas。"""
    fallback = None
    for p in FONTS:
        if not os.path.exists(p):
            continue
        try:
            f = ImageFont.truetype(p, px)
        except Exception:
            continue
        if fallback is None:
            fallback = f
        if (not need_cjk) or has_cjk(f):
            return f
    return fallback if fallback is not None else ImageFont.load_default()


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(65536), b""):
            h.update(blk)
    return h.hexdigest()[:12]


def render(txt_path, outdir, tail=None, title=None, note=None):
    text, enc = read_text_any(txt_path)
    lines = text.splitlines()
    lines = [l.rstrip("\r") for l in lines]
    if tail:
        lines = lines[-tail:]
    # 太长的行截断，保证图不会宽到没法插
    width_chars = 118
    lines = [l[:width_chars] + ("…" if len(l) > width_chars else "") for l in lines]

    fs = 15
    need_cjk = any("\u4e00" <= c <= "\u9fff" for c in "\n".join(lines) + txt_path)
    font = pick_font(fs, need_cjk)
    small = pick_font(12, need_cjk)
    lh = fs + 6
    pad = 18
    header_h = 30
    meta_h = 26

    probe = Image.new("RGB", (10, 10))
    pd = ImageDraw.Draw(probe)
    disp = lines + ([note] if note else [])
    maxw = [pd.textlength(l, font=font) for l in disp] + [pd.textlength("x" * 90, font=font)]
    w = int(min(max(maxw) + pad * 2, 1500))
    note_h = (lh + 8) if note else 0
    h = header_h + meta_h + note_h + len(lines) * lh + pad * 2

    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    # 标题栏
    d.rectangle([0, 0, w, header_h], fill=TITLE_BG)
    for i, col in enumerate([(222, 90, 86), (245, 198, 80), (98, 196, 140)]):
        d.ellipse([12 + i * 20, 10, 24 + i * 20, 22], fill=col)
    d.text((12 + 3 * 20 + 8, 8), title or os.path.relpath(txt_path, ROOT).replace("\\", "/"),
           font=small, fill=DIM)

    # 元信息行：证明这张图对应哪份文件、什么时候生成
    st = os.stat(txt_path)
    meta = "源文件 %s ｜ 编码 %s ｜ 最后写入 %s ｜ SHA1 %s ｜ 本图生成 %s" % (
        os.path.basename(txt_path), enc,
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
        sha1(txt_path), time.strftime("%Y-%m-%d %H:%M:%S"))
    d.text((pad, header_h + 6), meta, font=small, fill=DIM)

    y = header_h + meta_h
    if note:
        d.rectangle([pad, y, w - pad, y + lh + 4], fill=WARN_BG)
        d.text((pad + 8, y + 4), note, font=small, fill=WARN_FG)
        y += note_h
    for l in lines:
        col = FG
        if l.startswith("Traceback") or "Error" in l or "错误" in l:
            col = (232, 120, 112)
        elif l.startswith("[") or "OK" in l or "最好" in l or "Top-1" in l:
            col = ACC
        d.text((pad, y), l, font=font, fill=col)
        y += lh

    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(txt_path))[0]
    out = os.path.join(outdir, "证据_%s.png" % base)
    img.save(out)
    # 标注另存一份同名 .note.txt，check_figs.py 靠它判断"失败记录有没有被说明"
    side = out + ".note.txt"
    if note:
        with open(side, "w", encoding="utf-8") as fh:
            fh.write(note + "\n")
    elif os.path.exists(side):
        os.replace(side, side + ".bak")
    print("[OK] %s  (%d 行, %dx%d px)" % (out, len(lines), w, h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "证据"))
    ap.add_argument("--tail", type=int, default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--note", default=None, help="在图顶部加一条醒目标注，说明这份日志的性质")
    a = ap.parse_args()
    for p in a.logs:
        p = p if os.path.isabs(p) else os.path.join(ROOT, p)
        if os.path.exists(p):
            out = render(p, a.out, a.tail, a.title, a.note)
            body = read_text_any(p)[0]
            fails = [k for k in ("失败", "Error", "URLError", "Traceback", "ERROR REPORT") if k in body]
            if fails and not a.note:
                print("  [提醒] 这份日志含失败记录（%s），却没加 --note 说明性质，"
                      "放进论文容易被当成成功证据" % "、".join(fails))
        else:
            print("[跳过] 找不到 %s" % p)


if __name__ == "__main__":
    main()
