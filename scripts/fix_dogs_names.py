# -*- coding: utf-8 -*-
"""把 dogs 的类别目录统一成三位数（000..119），并同步 labels CSV。
原因：之前 0-99 用两位、100-119 用三位，ImageFolder 按目录名字典序分配类别索引，
      混宽会让 "10" < "100" < "11"，导致 ImageFolder 的类别序号 ≠ 原始 label 编号。
"""
import os, csv

ROOT = 'E:/PetRecognition/data/dogs'

for sp in ('train', 'test'):
    d = os.path.join(ROOT, sp)
    dirs = [x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]
    ren = 0
    for x in dirs:
        want = '%03d' % int(x)
        if want != x:
            os.rename(os.path.join(d, x), os.path.join(d, want))
            ren += 1
    print('dogs/%s：类别目录 %d 个，重命名 %d 个' % (sp, len(dirs), ren))

for f in ('labels_train.csv', 'labels_test.csv'):
    p = os.path.join(ROOT, f)
    with open(p, encoding='utf-8') as fh:
        rows = list(csv.reader(fh))
    head, body = rows[0], [r for r in rows[1:] if r]
    n = 0
    for r in body:
        parts = r[0].split(os.sep) if os.sep in r[0] else r[0].split('\\')
        want = '%03d' % int(parts[0])
        if want != parts[0]:
            parts[0] = want
            n += 1
        r[0] = '/'.join(parts)          # CSV 里统一用正斜杠，跨平台好读
        r[1] = '%03d' % int(r[1])       # breed 列也补成三位，方便对照
    with open(p, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['file', 'breed'])
        w.writerows(body)
    print('%s：改写 %d 行，共 %d 行' % (f, n, len(body)))

# 校验：字典序 == 数值序
for sp in ('train', 'test'):
    d = os.path.join(ROOT, sp)
    names = sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))
    nums = [int(x) for x in names]
    ok = nums == list(range(len(nums)))
    print('校验 dogs/%s：目录数 %d，字典序与数值序 %s，范围 %s..%s'
          % (sp, len(names), '一致 ✓' if ok else '不一致 ✗', names[0], names[-1]))
