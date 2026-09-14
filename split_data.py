import os
from shutil import copy, rmtree
import random

BASE_DIR = r'C:\Users\26281\kc_dog_and_cat'
file_path = os.path.join(BASE_DIR, 'data')

# 获取data文件夹下所有子文件夹（原始类别：cat、dog）
flower_class = [cla for cla in os.listdir(file_path)
                if os.path.isdir(os.path.join(file_path, cla))]

print(f"检测到原始类别: {flower_class}")

# 清理旧的 train/val（绝对路径，避免脏目录）
train_dir = os.path.join(file_path, 'train')
val_dir = os.path.join(file_path, 'val')
if os.path.exists(train_dir):
    rmtree(train_dir)
if os.path.exists(val_dir):
    rmtree(val_dir)

# 创建 train 和 val 目录（绝对路径）
for cla in flower_class:
    os.makedirs(os.path.join(train_dir, cla), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cla), exist_ok=True)

split_rate = 0.2

for cla in flower_class:
    cla_path = os.path.join(file_path, cla)
    images = [img for img in os.listdir(cla_path)
              if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    num = len(images)
    eval_index = random.sample(images, k=int(num * split_rate))

    for index, image in enumerate(images):
        src = os.path.join(cla_path, image)
        if image in eval_index:
            dst = os.path.join(val_dir, cla, image)
        else:
            dst = os.path.join(train_dir, cla, image)
        copy(src, dst)
        print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")
    print()

print("\nprocessing done!")
for cla in flower_class:
    t = len(os.listdir(os.path.join(train_dir, cla)))
    v = len(os.listdir(os.path.join(val_dir, cla)))
    print(f"  {cla}: train={t}, val={v}")