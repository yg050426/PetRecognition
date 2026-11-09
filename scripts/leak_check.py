# -*- coding: utf-8 -*-
"""推 GitHub 前的身份泄露扫描。

仓库是公开的，所以这个脚本本身**不能**把要查的词写死在里面——否则它自己就成了泄露源。
敏感词放在同目录的 leak_patterns.txt（已加进 .gitignore，不进仓库），
脚本里只留通用规则：邮箱、手机号、身份证号、带用户名的本机绝对路径。

用法：
    python scripts/leak_check.py            # 扫 git 会提交的文件
    python scripts/leak_check.py --all      # 连未跟踪文件一起扫（提交前用这个）
"""
import os
import re
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PFILE = os.path.join(ROOT, "scripts", "leak_patterns.txt")

# 通用规则：不依赖任何个人信息，写在脚本里也安全
BASE = [
    ("邮箱", r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}"),
    ("手机号", r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    ("身份证号", r"(?<!\d)\d{17}[\dXx](?!\d)"),
    ("学号样式", r"(?<!\d)20\d{8}(?!\d)"),
    ("本机用户目录", r"[Cc]:[\\/]+[Uu]sers[\\/]+[A-Za-z0-9_.-]{3,}"),
    ("网盘/IM 本地目录", r"[A-Za-z]:[\\/]+(WeChat|Weixin|xwechat_files|Tencent|QQ)"),
]

SKIP_EXT = {".pth", ".jpg", ".jpeg", ".png", ".parquet", ".zip", ".gz", ".tgz",
            ".onnx", ".bin", ".npy", ".npz", ".ico", ".pdf"}
TEXT_EXT = {".py", ".md", ".js", ".ps1", ".txt", ".csv", ".json", ".yml", ".yaml",
            ".ipynb", ".sh", ".bat", ".cfg", ".ini", ""}


def patterns():
    out = list(BASE)
    if os.path.exists(PFILE):
        for line in open(PFILE, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line or line.startswith("#") or "\t" not in line:
                continue
            name, pat = line.split("\t", 1)
            out.append((name, pat))
    else:
        print("[警告] 没有 %s，只跑了通用规则；个人身份词（姓名/学校/班级/老师称谓）没查。"
              % os.path.relpath(PFILE, ROOT))
    return out


def git_files(all_mode):
    """列出将要提交的全部内容 = 已跟踪 + 未跟踪且未被忽略。

    两个坑都在这：
      · 必须用 -z。Windows 下 git 会把非 ASCII 文件名转成 "\347\225\231..." 八进制转义，
        按行 split 得到的是不存在的路径，中文命名的文件会被静默跳过（我踩过一次）。
      · 必须分两次列。`git ls-files --others` 只给未跟踪文件，
        只跑它会漏掉全部已跟踪文件（我踩过第二次，扫描数从 38 掉到 19 才发现）。
    """
    out = []
    runs = [["git", "ls-files", "-z"]]
    if all_mode:
        runs.append(["git", "ls-files", "-z", "--others", "--exclude-standard"])
    for args in runs:
        r = subprocess.run(args, cwd=ROOT, capture_output=True)
        out += [f.decode("utf-8", "replace") for f in r.stdout.split(b"\0") if f]
    # 去重保序
    seen, uniq = set(), []
    for f in out:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


def main():
    all_mode = "--all" in sys.argv
    pats = patterns()
    files = git_files(all_mode)
    hits, scanned, skipped = [], 0, []
    for rel in files:
        p = os.path.join(ROOT, rel.replace("/", os.sep))
        if not os.path.isfile(p):
            skipped.append(rel)
            continue
        ext = os.path.splitext(p)[1].lower()
        if ext in SKIP_EXT:
            continue
        if ext and ext not in TEXT_EXT:
            continue
        try:
            text = open(p, encoding="utf-8").read()
        except UnicodeDecodeError:
            try:
                text = open(p, encoding="gbk").read()
            except Exception:
                skipped.append(rel + "（编码不明）")
                continue
        scanned += 1
        for ln, line in enumerate(text.splitlines(), 1):
            for name, pat in pats:
                m = re.search(pat, line)
                if m:
                    hits.append((rel, ln, name, line.strip()[:100]))

    print("扫描 %d 个文本文件（共列出 %d 项，跳过 %d 项）；规则 %d 条"
          % (scanned, len(files), len(skipped), len(pats)))
    for key in ("user.name", "user.email"):
        r = subprocess.run(["git", "config", key], cwd=ROOT, capture_output=True, text=True)
        v = r.stdout.strip()
        print("git config %s = %s" % (key, v or "(未设置)"))
        if key == "user.email" and re.search(r"^\d+@", v):
            print("   [警告] noreply 邮箱带数字前缀＝QQ 号，要用不带数字前缀的那种写法")

    if not hits:
        print("\n没发现身份泄露。")
        return
    by_file = {}
    for rel, ln, name, line in hits:
        by_file.setdefault(rel, []).append((ln, name, line))
    print("\n发现 %d 处可疑，分布在 %d 个文件：" % (len(hits), len(by_file)))
    for rel, rows in sorted(by_file.items()):
        print("\n■ %s" % rel)
        for ln, name, line in rows[:15]:
            print("   行%-5d [%s] %s" % (ln, name, line))
        if len(rows) > 15:
            print("   ...还有 %d 处" % (len(rows) - 15))
    sys.exit(1)


if __name__ == "__main__":
    main()
