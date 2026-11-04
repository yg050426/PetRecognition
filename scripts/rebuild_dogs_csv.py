# -*- coding: utf-8 -*-
"""以磁盘为准，重建 dogs 的 labels CSV 与 class_names（synset 编号）表。
背景：dogs 类别目录已从混宽(00..99,100..119)统一为三位数(000..119)，
      旧 CSV 里的前缀和文件名已对不上，这里按实际文件重新生成，保证一致。
"""
import os, csv

ROOT = 'E:/PetRecognition/data/dogs'


def rebuild(sp):
    d = os.path.join(ROOT, sp)
    rows = []
    for cls in sorted(os.listdir(d)):
        p = os.path.join(d, cls)
        if not os.path.isdir(p):
            continue
        for fn in sorted(os.listdir(p)):
            if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                rows.append(['%s/%s' % (cls, fn), '%03d' % int(cls)])
    out = os.path.join(ROOT, 'labels_%s.csv' % sp)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['file', 'breed'])
        w.writerows(rows)
    print('  labels_%s.csv 重写 %d 行' % (sp, len(rows)))
    return rows


rows = []
for sp in ('train', 'test'):
    rows += rebuild(sp)

# 从文件名里提取 synset（n021xxxxx），生成 120 类的编号-合成名对照
syn = {}
for r in rows:
    fn = r[0].split('/')[1]
    for tok in fn.replace('_', '/').split('/'):
        if tok.startswith('n0'):
            syn.setdefault(r[1], tok)
            break
missing = [c for c in sorted({r[1] for r in rows}) if c not in syn]
with open(os.path.join(ROOT, 'class_synsets.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['breed_id', 'synset'])
    for k in sorted(syn):
        w.writerow([k, syn[k]])
print('  class_synsets.csv 写了 %d 类，缺 synset 的 %d 类' % (len(syn), len(missing)))

# 校验：目录数、图片数、字典序==数值序
for sp in ('train', 'test'):
    d = os.path.join(ROOT, sp)
    dirs = sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))
    nums = [int(x) for x in dirs]
    total = sum(len(os.listdir(os.path.join(d, x))) for x in dirs)
    print('  校验 dogs/%s：类别 %d，图片 %d，字典序与数值序 %s'
          % (sp, len(dirs), total, '一致' if nums == list(range(len(nums))) else '不一致'))
