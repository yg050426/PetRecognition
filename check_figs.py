# -*- coding: utf-8 -*-
"""留图自检：把"打开看图"这件事变成一条命令。

查四件真会出事的事：
  1) 方块字——图里画不出来的字，根源只有两个：读文件时编码选错，或者字体没有这个字形。
     编码用"能不能严格解码"判；字形用 fontTools 直接读字体的 cmap 表判，
     不靠"画出来有没有像素"——缺字形时 PIL 会画一个空心方块，那是有像素的，判不出来。
  2) 报错日志冒充实验证据——日志里只有 traceback、没有结果行，这种图不能交。
  3) 图注清单和实际文件对不对得上（有没有漏画、多画、编号撞车）。
  4) 成图用的中文在 matplotlib 选中的字体里是不是都有字形。

用法：
    python check_figs.py            # 全查，有问题就非零退出
    python check_figs.py --verbose  # 把每张图的判定依据都打出来
"""
import argparse
import glob
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(ROOT, "figures")
EVID = os.path.join(FIGS, "证据")

CJK = re.compile(r"[一-鿿 -〿＀-￯]")

# 日志里"这是有效输出"和"这只是报错"的特征
OK_MARKS = ("Top-1", "Top1", "val_top1", "ep01", "ep0", "训练", "下载完成", "NVIDIA", "MiB")
ERR_MARKS = (">>>>>>>>>>>>>>>>>>>> ERROR REPORT", "Traceback (most recent call last)",
             "conda --no-plugins", "CONDA_NO_PLUGINS")
# 只要出现这些词，说明日志里记的是失败过程，必须挂 --note 说明清楚
FAIL_MARKS = ("失败", "Error", "URLError", "Traceback", "ERROR REPORT")


def detect_enc(raw):
    for e in ("utf-8-sig", "utf-8", "gbk", "cp936"):
        try:
            return raw.decode(e), e
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8(有坏字节)"


def cmap_of(path, index=0):
    """读出字体真正带有的码位集合。"""
    from fontTools.ttLib import TTFont
    f = TTFont(path, fontNumber=index, lazy=True)
    cm = f.getBestCmap()
    f.close()
    return set(cm.keys())


def cjk_chars(text):
    return {c for c in text if ord(c) > 0x2E80 and not c.isascii()}


def check_evidence(verbose=False):
    problems = []
    rows = []
    for png in sorted(glob.glob(os.path.join(EVID, "*.png"))):
        name = os.path.basename(png)
        # 证据图文件名是 证据_<日志名>.png，反推源文件
        base = name[len("证据_"):-len(".png")]
        src = None
        for ext in (".log", ".txt", ""):
            cand = os.path.join(ROOT, "logs", base + ext)
            if os.path.exists(cand):
                src = cand
                break
        if src is None:
            problems.append("%s：找不到对应的源日志" % name)
            continue

        text, enc = detect_enc(open(src, "rb").read())
        bad = text.count("\ufffd")
        chars = cjk_chars(text)

        # 字体能不能覆盖这页里出现的每个汉字
        sys.path.insert(0, ROOT)
        import log2fig
        font_path = None
        for p in log2fig.FONTS:
            if os.path.exists(p):
                try:
                    idx = 0
                    cm = cmap_of(p, idx)
                except Exception:
                    continue
                if 0x4E2D in cm:      # 有"中"字才算带中文
                    font_path = p
                    break
        missing = sorted(c for c in chars if ord(c) not in cmap_of(font_path)) if (font_path and chars) else []

        has_ok = any(m in text for m in OK_MARKS)
        has_err = any(m in text for m in ERR_MARKS)
        if has_err and not has_ok:
            kind = "只有报错"
        elif has_err and has_ok:
            kind = "报错+结果混合"
        else:
            kind = "有效输出"

        rows.append((name, os.path.relpath(src, ROOT).replace("\\", "/"), enc,
                     len(text.splitlines()), kind, len(chars), len(missing), bad))
        if bad:
            problems.append("%s：解码后有 %d 个替换字符（编码判错）" % (name, bad))
        if missing:
            problems.append("%s：字体缺 %d 个字形，会画成方块：%s" %
                            (name, len(missing), "".join(missing[:12])))
        if kind == "只有报错":
            problems.append("%s：源日志只有报错、没有结果，不能当实验证据（重跑时别用 conda run）" % name)
        fails = [k for k in FAIL_MARKS if k in text]
        has_note = os.path.exists(png + ".note.txt")
        if fails and not has_note:
            problems.append("%s：日志里有失败记录（%s）却没有 .note.txt 标注，"
                            "容易被当成成功证据；重跑 log2fig.py 时加 --note" % (name, "、".join(fails)))
        if fails and has_note:
            note = open(png + ".note.txt", encoding="utf-8").read().strip()
            if not note:
                problems.append("%s：.note.txt 是空的" % name)
        if verbose:
            print("  %-34s 源=%-28s 编码=%-10s 行数=%-4s %s 汉字种类=%-3d 缺字形=%d 替换符=%d"
                  % rows[-1])
    return rows, problems


def check_paper_figs(verbose=False):
    problems = []
    pngs = sorted(glob.glob(os.path.join(FIGS, "*.png")))
    # 图和表是两条独立编号，所以 key 要带上"图"/"表"前缀，否则 图5-01 和 表5-01 会被误判撞车
    nums = [re.match(r"([图表])(\d+-\d+)", os.path.basename(p)) for p in pngs]
    seen = {}
    for p, m in zip(pngs, nums):
        if not m:
            problems.append("%s：文件名不符合 图5-XX_/表5-XX_ 规则" % os.path.basename(p))
            continue
        seen.setdefault(m.group(1) + m.group(2), []).append(os.path.basename(p))
    for k, v in seen.items():
        if len(v) > 1:
            problems.append("编号 %s 撞车，有 %d 个文件：%s" % (k, len(v), ", ".join(v)))

    # 成图脚本里所有中文，在 matplotlib 选中的字体里是否都有字形
    src = open(os.path.join(ROOT, "make_figs.py"), encoding="utf-8").read()
    chars = cjk_chars(src)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import make_figs
    make_figs.setup_font()
    name = matplotlib.rcParams["font.sans-serif"][0]
    path = font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
    idx = 1 if path.lower().endswith("msyh.ttc") else 0
    cm = cmap_of(path, idx) if os.path.exists(path) else set()
    missing = sorted(c for c in chars if ord(c) not in cm)
    if verbose:
        print("  成图 %d 张，编号无重复=%s；matplotlib 字体=%s (%s)；中文 %d 种，缺字形 %d"
              % (len(pngs), not any("撞车" in p for p in problems), name, os.path.basename(path),
                 len(chars), len(missing)))
    if missing:
        problems.append("成图字体 %s 缺 %d 个字形：%s" % (os.path.basename(path), len(missing), "".join(missing[:12])))
    return problems


def check_manifest():
    problems = []
    mf = os.path.join(FIGS, "图注清单.md")
    if not os.path.exists(mf):
        return ["图注清单.md 不存在，跑一遍 make_figs.py"]
    txt = open(mf, encoding="utf-8").read()
    listed = set(re.findall(r"`([^`]+\.png)`", txt))
    actual = {os.path.basename(p) for p in glob.glob(os.path.join(FIGS, "*.png"))}
    for miss in sorted(actual - listed):
        problems.append("%s 在磁盘上但清单里没有" % miss)
    for ghost in sorted(listed - actual):
        problems.append("清单里写着 %s 但磁盘上没有（清单过期，重跑 make_figs.py）" % ghost)
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    print("【1】证据图：编码 / 字形 / 是不是只有报错")
    rows, probs = check_evidence(a.verbose)
    if not a.verbose:
        for r in rows:
            print("  %-34s 编码=%-10s %s" % (r[0], r[2], r[4]))
    print("\n【2】论文成图：编号 / 字体覆盖")
    probs += check_paper_figs(a.verbose)
    print("\n【3】图注清单与磁盘是否一致")
    mp = check_manifest()
    probs += mp

    print("\n" + "=" * 62)
    if probs:
        print("发现 %d 个问题：" % len(probs))
        for p in probs:
            print("  ✗ " + p)
        sys.exit(1)
    print("全部通过：没有方块字风险，没有报错日志冒充证据，编号和清单一致。")


if __name__ == "__main__":
    main()
