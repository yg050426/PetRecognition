# -*- coding: utf-8 -*-
"""把下载好的 parquet 解成 ImageFolder 目录结构，供 PyTorch 直接训练。

产出：
  data/pet/train/00/*.jpg ... data/pet/train/36/*.jpg     37 个品种
  data/pet/test/...
  data/pet/labels.csv        文件名, 品种编号, 物种(0=猫?按原数据), 原始image_id
  data/dogs/train/000/*.jpg  120 个犬种
  data/dogs/test/...
  data/dogs/labels.csv
"""
import os, csv, time
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")


def write_images(path_list, label_list, extra_list, outdir, prefix):
    os.makedirs(outdir, exist_ok=True)
    rows = []
    n = 0
    t0 = time.time()
    for i, (img, lab, extra) in enumerate(zip(path_list, label_list, extra_list)):
        b = img.get("bytes") if isinstance(img, dict) else None
        if not b:
            continue
        cls = "%02d" % int(lab) if int(lab) < 100 else "%03d" % int(lab)
        d = os.path.join(outdir, cls)
        os.makedirs(d, exist_ok=True)
        raw = (img.get("path") or "").replace("\\", "/").split("/")[-1]
        if not raw:
            raw = "%s_%06d.jpg" % (prefix, i)
        fname = "%s_%s" % (cls, raw)
        with open(os.path.join(d, fname), "wb") as f:
            f.write(b)
        rows.append([os.path.join(cls, fname), lab, extra])
        n += 1
        if n % 2000 == 0:
            print("   ... %d 张 (%.0fs)" % (n, time.time() - t0), flush=True)
    return rows, n


def unpack_pet(split):
    t = pq.read_table(os.path.join(DATA, "raw", "pet_%s.parquet" % split))
    cols = t.column("image").to_pylist()
    labs = t.column("label").to_pylist()
    ids = t.column("image_id").to_pylist()
    cats = t.column("label_cat_dog").to_pylist()
    out = os.path.join(DATA, "pet", split)
    rows, n = write_images(cols, [l for l in labs], [str(c) for c in cats], out, "pet")
    with open(os.path.join(DATA, "pet", "labels_%s.csv" % split), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "breed", "cat_dog", "image_id"])
        for r, iid in zip(rows, ids):
            w.writerow(r + [iid])
    print("[pet/%s] 写出 %d 张" % (split, n), flush=True)


def unpack_dogs(name):
    t = pq.read_table(os.path.join(DATA, "raw", "%s.parquet" % name))
    cols = t.column("pixel_values").to_pylist()
    labs = t.column("label").to_pylist()
    split = "test" if "test" in name else "train"
    out = os.path.join(DATA, "dogs", split)
    rows, n = write_images(cols, labs, [""] * len(labs), out, "dog")
    tag = "labels_%s.csv" % split
    p = os.path.join(DATA, "dogs", tag)
    new = not os.path.exists(p)
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "breed"])
        w.writerows(rows)
    print("[dogs/%s 来自 %s] 写出 %d 张" % (split, name, n), flush=True)


if __name__ == "__main__":
    print("== Pet trainval(=train) ==", flush=True)
    unpack_pet("train")
    print("== Pet test ==", flush=True)
    unpack_pet("test")
    for nm in ["dogs_train1", "dogs_train2", "dogs_test"]:
        print("== %s ==" % nm, flush=True)
        unpack_dogs(nm)
    print("全部解包完成。", flush=True)
