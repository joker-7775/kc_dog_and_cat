import os
import torch
import torchvision.models as models
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder

ROOT_TEST = r'C:/Users/26281/kc_dog_and_cat/data/val'
MODEL_PATH = r'C:/Users/26281/kc_dog_and_cat/save_model/best_model.pth'

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

# 用于显示的反归一化
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)
print(f"验证集大小: {len(val_dataset)}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 重建 ResNet18 模型（和训练时一致）
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model = model.to(device)
model.eval()

classes = ["cat", "dog"]

with torch.no_grad():
    correct = 0
    total = len(val_dataset)

    for i in range(total):
        x, y = val_dataset[i][0], val_dataset[i][1]
        x_input = torch.unsqueeze(x, dim=0).to(device)
        pred = model(x_input)
        pred_idx = torch.argmax(pred[0]).item()

        if pred_idx == y:
            correct += 1

    print(f"\n整体准确率: {correct}/{total} = {correct/total*100:.1f}%")

print("\n---------------- 随机测试 5 张图片 ----------------")
import random
indices = random.sample(range(len(val_dataset)), min(5, len(val_dataset)))

with torch.no_grad():
    for idx in indices:
        x, y = val_dataset[idx][0], val_dataset[idx][1]

        img_restore = x * std + mean
        img_np = img_restore.permute(1, 2, 0).cpu().numpy()
        img_np = img_np.clip(0, 1)

        x_input = torch.unsqueeze(x, dim=0).to(device)
        pred = model(x_input)
        pred_idx = torch.argmax(pred[0]).item()

        predicted = classes[pred_idx]
        actual = classes[y]
        result = "√" if predicted == actual else "×"

        plt.figure()
        plt.imshow(img_np)
        plt.title(f"{result} Predicted:{predicted}, Actual:{actual}")
        plt.axis("off")
        plt.show()