import torch
from torch import nn
import torchvision.models as models
import numpy as np
from torch.optim import lr_scheduler
import os

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT_TRAIN = r'C:/Users/26281/kc_dog_and_cat/data/train'
ROOT_TEST = r'C:/Users/26281/kc_dog_and_cat/data/val'

# ============== 1. 更丰富的数据增强 ==============
# ImageNet 预训练模型专用的均值和标准差
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    normalize
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")

# 小数据集用更小的 batch_size，让梯度更新更频繁
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# ============== 2. 使用预训练 ResNet18（迁移学习） ==============
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 冻结前面的层（小数据集强烈建议先冻结，效果更好）
for param in model.parameters():
    param.requires_grad = False

# 替换最后的全连接层，猫狗二分类
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

# 打印可训练参数
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"可训练参数: {trainable_params}/{total_params}")

# ============== 3. 损失函数 + Adam优化器 + CosineAnnealing学习率调度 ==============
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=1e-4)
lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    loss, current, n = 0.0, 0.0, 0
    for batch, (x, y) in enumerate(dataloader):
        image, y = x.to(device), y.to(device)
        output = model(image)
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis=1)
        cur_acc = torch.sum(y == pred) / output.shape[0]

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss.item()
        current += cur_acc.item()
        n += 1
    train_loss = loss / n
    train_acc = current / n
    print(f'train_loss: {train_loss:.4f}')
    print(f'train_acc: {train_acc:.4f}')
    return train_loss, train_acc


def val(dataloader, model, loss_fn):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            image, y = x.to(device), y.to(device)
            output = model(image)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n += 1

    val_loss = loss / n
    val_acc = current / n
    print(f'val_loss: {val_loss:.4f}')
    print(f'val_acc: {val_acc:.4f}')
    return val_loss, val_acc


def matplot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("训练集和验证集loss值对比图")
    plt.show()


def matplot_acc(train_acc, val_acc):
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title("训练集和验证集acc值对比图")
    plt.show()


# ============== 4. 增加 epoch + 早停机制 ==============
loss_train = []
acc_train = []
loss_val = []
acc_val = []

epoch = 100
min_acc = 0
patience = 20
patience_counter = 0
best_epoch = 0

for t in range(epoch):
    print(f"\nepoch {t+1}/{epoch}  lr={optimizer.param_groups[0]['lr']:.6f}\n--------------")
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
    val_loss, val_acc = val(val_dataloader, model, loss_fn)
    lr_scheduler.step()

    loss_train.append(train_loss)
    acc_train.append(train_acc)
    loss_val.append(val_loss)
    acc_val.append(val_acc)

    # 保存最好的模型权重
    if val_acc > min_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir('save_model')
        min_acc = val_acc
        best_epoch = t + 1
        patience_counter = 0
        print(f"★ save best model, 第{t+1}轮, val_acc={val_acc:.4f}")
        torch.save(model.state_dict(), folder + '/' + 'best_model.pth')
    else:
        patience_counter += 1
        print(f"第{t+1}轮, val_acc未提升 ({patience_counter}/{patience})")

    # 早停: 防止过拟合
    if patience_counter >= patience:
        print(f"\n早停触发! 第{t+1}轮停止训练")
        print(f"最佳轮次: 第{best_epoch}轮, 最佳val_acc: {min_acc:.4f}")
        break

    # 保存最后一轮
    if t == epoch - 1:
        torch.save(model.state_dict(), folder + '/' + 'last_model.pth')

matplot_loss(loss_train, loss_val)
matplot_acc(acc_train, acc_val)
print(f'\nDone! 最佳验证准确率: {min_acc:.4f} (第{best_epoch}轮)')