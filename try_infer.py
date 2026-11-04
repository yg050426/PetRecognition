# -*- coding: utf-8 -*-
"""验证环境：显卡 + PyTorch + 预训练模型 是否真的能跑通。
图片准备了好几个备用地址，全部失败时改用合成图（结果无意义，但足以证明 GPU 前向通路正常）。
"""
import os, urllib.request
import torch, torchvision
from torchvision import transforms
from PIL import Image

IDS = {151: "大麦町", 152: "澳洲野犬", 153: "阿富汗猎犬", 154: "巴吉度", 155: "比格", 156: "寻血猎犬",
       157: "布列塔尼", 158: "柯基", 159: "松狮", 160: "荷兰狮毛犬", 161: "刚毛指示犬", 162: "彭布罗克",
       198: "虎斑猫", 199: "石纹猫", 200: "波斯猫", 201: "暹罗猫", 202: "埃及猫", 203: "美洲狮",
       204: "猞猁", 205: "豹", 206: "雪豹", 207: "美洲虎", 208: "狮", 209: "虎",
       281: "金毛寻回犬", 282: "拉布拉多", 283: "可卡犬", 284: "布列塔尼犬", 285: "克鲁犬",
       287: "贵宾犬", 288: "墨西哥无毛犬", 289: "灰狼"}

URLS = [
    "https://cdn.jsdelivr.net/gh/pytorch/hub@master/images/dog.jpg",
    "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
    "https://gitee.com/mirrors/pytorch_hub/raw/master/images/dog.jpg",
]


def get_image():
    for u in URLS:
        try:
            print("  尝试取测试图：%s" % u.split("/")[2], flush=True)
            urllib.request.urlretrieve(u, "dog.jpg")
            if os.path.getsize("dog.jpg") > 5000:
                return "dog.jpg", True
        except Exception as e:
            print("    失败：%s" % type(e).__name__, flush=True)
    # 兜底：画一张合成图，只为验证 GPU 通路
    print("  三个地址都不通，改用合成图（分类结果无意义，但能验证通路）", flush=True)
    im = Image.new("RGB", (320, 320), (180, 140, 90))
    im.save("dog.jpg")
    return "dog.jpg", False


dev = "cuda" if torch.cuda.is_available() else "cpu"
print("PyTorch：%s   torchvision：%s" % (torch.__version__, torchvision.__version__))
print("运行设备：%s" % dev, end="")
print("  (%s)" % torch.cuda.get_device_name(0) if dev == "cuda" else "  (CPU，训练会非常慢，需解决)")

print("\n[1/3] 加载 ResNet50 预训练权重（首次约 100MB，从 download.pytorch.org）", flush=True)
model = torchvision.models.resnet50(weights="IMAGENET1K_V1").eval().to(dev)
print("  权重加载成功，模型已放到 %s" % dev, flush=True)

print("\n[2/3] 取测试图片", flush=True)
path, real = get_image()

print("\n[3/3] 前向推理", flush=True)
img = Image.open(path).convert("RGB")
pre = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
import time
x = pre(img).unsqueeze(0).to(dev)
with torch.no_grad():
    model(x)                      # 预热
    t0 = time.time()
    out = model(x)[0]
    torch.cuda.synchronize() if dev == "cuda" else None
    ms = (time.time() - t0) * 1000
top = out.topk(5)
print("  单张推理耗时：%.1f ms（含一次预热后的稳态）" % ms, flush=True)
print("\nTop-5：", flush=True)
for i, (s, idx) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
    print("  %d) 编号 %4d  得分 %6.3f  %s" % (i, idx, s, IDS.get(idx, "(非犬猫类)")), flush=True)

print("\n结论：%s" % ("环境完全可用，可以开始训练了" if dev == "cuda" else "CUDA 没启用，需要先解决显卡问题"), flush=True)
if not real:
    print("（注：本次用的是合成图，类别结果无意义，仅验证通路）", flush=True)
