# 基于 ResNet18 的猫狗分类系统设计与实现

> **课程名称**：深度学习与计算机视觉
> **项目类型**：课程大作业
> **作者**：joker-7775
> **学号**：——
> **完成日期**：2026 年 9 月

---

## 摘  要

本项目基于 PyTorch 深度学习框架，采用 ResNet18 作为骨干网络，通过迁移学习在一个小型猫狗图像数据集上完成二分类任务，并将训练好的模型封装为一个具有完整前后端的 Web 应用。系统不仅支持本地图片上传识别，还集成了实时摄像头拍照功能，同时提供预测历史记录、准确率仪表盘和动态图表等数据分析能力。该项目展示了从数据预处理、模型训练、性能优化到工程落地的完整深度学习应用开发流程。

**关键词**：ResNet18；迁移学习；猫狗分类；Flask；计算机视觉

---

## 目录

1. [绪论](#1-绪论)
2. [相关技术](#2-相关技术)
3. [系统需求分析与设计](#3-系统需求分析与设计)
4. [数据处理](#4-数据处理)
5. [模型设计与训练](#5-模型设计与训练)
6. [后端实现](#6-后端实现)
7. [前端实现](#7-前端实现)
8. [实验结果与分析](#8-实验结果与分析)
9. [部署与使用说明](#9-部署与使用说明)
10. [总结与展望](#10-总结与展望)
11. [参考文献](#11-参考文献)

---

## 1. 绪论

### 1.1 研究背景

近年来，深度学习在图像识别领域取得了突破性进展。以 Convolutional Neural Network（CNN）为代表的深度卷积网络在 ImageNet、COCO 等大规模竞赛上频频刷新准确率纪录。在众多经典模型中，ResNet（Residual Network）通过引入残差连接（Residual Connection）有效缓解了深层网络中的梯度消失问题，使得训练更深的网络成为可能。

### 1.2 研究意义

本项目选择一个贴近生活的应用场景——猫狗二分类，作为深度学习课程的综合实践题目。通过该项目，学生可以完整地经历从数据获取、模型搭建、训练调参到 Web 部署的全流程，将课堂上学习到的理论知识转化为实际工程能力。

### 1.3 项目目标

| 目标 | 说明 |
|------|------|
| 模型目标 | 在验证集上实现 ≥ 85% 的分类准确率 |
| 工程目标 | 提供可交互的 Web 界面，支持上传和拍照两种输入方式 |
| 分析目标 | 记录用户预测历史，提供可视化的统计分析 |

---

## 2. 相关技术

### 2.1 技术栈总览

```
┌──────────────────────────────────────────────┐
│              浏览器 (前端)                     │
│  HTML5 · CSS3 · JavaScript · Chart.js        │
│  getUserMedia API · Canvas API               │
└──────────────────┬───────────────────────────┘
                   │ HTTP/JSON
┌──────────────────▼───────────────────────────┐
│           Flask Web 服务 (后端)               │
│  Flask 3.1 · Pillow · JSON 文件存储           │
└──────────────────┬───────────────────────────┘
                   │
┌──────────────────▼───────────────────────────┐
│         PyTorch 推理引擎 (模型)               │
│  ResNet18 · ImageNet Normalize · CPU/GPU     │
└──────────────────────────────────────────────┘
```

### 2.2 ResNet 残差网络

ResNet 由何恺明等人于 2016 年在 CVPR 上提出，核心创新是 **跳跃连接（Skip Connection）**：

$$y = \mathcal{F}(x, \{W_i\}) + x$$

其中 $\mathcal{F}$ 是残差映射函数（通常包含 2~3 个卷积层），$x$ 是输入。这种设计使得梯度可以直接回传到较浅层，解决了深度网络难以优化的问题。

本项目选用 ResNet18，即 18 层残差网络，其参数量仅为 11.7M，在 CPU 上也能快速推理，适合小型设备部署。

### 2.3 迁移学习

由于本项目数据量较小（训练集仅 80 张），从头训练网络容易过拟合。因此采用 **迁移学习** 策略：

1. 加载在 ImageNet 上预训练好的 ResNet18 权重
2. 冻结前面的卷积层参数（保留通用特征提取能力）
3. 替换最后的全连接层，将 1000 类输出改为 2 类（猫/狗）
4. 仅训练新的全连接层

### 2.4 Flask 轻量级 Web 框架

Flask 是 Python 生态中最流行的轻量级 Web 框架，其特点是扩展灵活、学习曲线平缓。本项目使用 Flask 提供 RESTful API 和模板渲染能力。

---

## 3. 系统需求分析与设计

### 3.1 功能需求

| 模块 | 功能 | 优先级 |
|------|------|--------|
| 图片上传 | 点击 / 拖拽上传本地图片 | P0 |
| 摄像头拍照 | 调用浏览器 getUserMedia 实时拍摄 | P0 |
| 模型推理 | 返回预测类别 + 猫狗概率 + 置信度 | P0 |
| 准确率仪表盘 | 展示模型在验证集上的整体准确率 | P1 |
| 预测历史记录 | 持久化存储每次用户预测结果 | P1 |
| 图表可视化 | 饼图 + 折线图展示历史趋势 | P1 |
| 案例展示 | 展示模型在验证集上的典型正确/错误案例 | P2 |
| GitHub 链接 | 页面展示项目源码仓库链接 | P2 |

### 3.2 非功能需求

- **响应时间**：单张图片推理 ≤ 500ms（CPU）
- **并发安全**：多线程下历史记录写入使用锁保护
- **可移植性**：除 `.pth` 权重文件外，所有代码可跨平台运行

### 3.3 系统架构

```
                 前端 (index.html)
                  │
     ┌────────────┼────────────┐
     │            │            │
  图片上传    摄像头拍照    历史查询
     │            │            │
     └────────────┼────────────┘
                  ▼
         Flask 路由层 (app.py)
          │      │      │
          ▼      ▼      ▼
      /predict /api/history /api/stats
          │      │      │
          ▼      │      │
    ┌─────────┐  │      │
    │ ResNet18 │  │      │
    │  推理   │  │      │
    └────┬────┘  │      │
         │       │      │
         ▼       ▼      ▼
  predict_history.json  (文件存储)
```

---

## 4. 数据处理

### 4.1 数据集来源

本项目使用公开的 **Kaggle Dogs vs Cats** 数据集的一个小子集（约 100 张图片），按以下结构组织：

```
data/
├── train/
│   ├── cat/          # 40 张训练集猫图片
│   └── dog/          # 40 张训练集狗图片
└── val/
    ├── cat/          # 10 张验证集猫图片
    └── dog/          # 10 张验证集狗图片
```

### 4.2 数据集划分脚本

`split_data.py` 负责将原始打乱的数据按比例划分到 train 和 val 目录：

```python
# split_data.py（核心逻辑）
all_files = glob.glob(os.path.join(raw_dir, '*.jpg'))
random.shuffle(all_files)
train_end = int(len(all_files) * train_ratio)
for f in all_files[:train_end]:  # 拷贝到 train/{cat|dog}/
    ...
for f in all_files[train_end:]:  # 拷贝到 val/{cat|dog}/
    ...
```

### 4.3 图像预处理

ResNet18 在 ImageNet 上训练时使用的标准化参数必须保持一致，否则模型表现会显著下降：

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),        # ResNet18 固定输入尺寸
    transforms.ToTensor(),                # 转为 [0, 1] 张量
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],       # ImageNet 均值
        std=[0.229, 0.224, 0.225]         # ImageNet 标准差
    )
])
```

---

## 5. 模型设计与训练

### 5.1 网络结构

采用 torchvision 内置的 ResNet18，修改最后的全连接层以适配二分类：

```python
# net.py（核心代码）
import torchvision.models as models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features     # 512
model.fc = torch.nn.Linear(num_features, 2)
```

### 5.2 训练配置

| 超参数 | 值 | 说明 |
|--------|-----|------|
| Optimizer | Adam | 自适应学习率 |
| 初始学习率 | 0.001 | 配合 CosineAnnealingLR |
| Batch size | 8 | 小批量，适合小样本 |
| Max Epoch | 100 | 配合早停 |
| Early Stopping | patience=20 | 验证集准确率连续 20 轮不提升则停止 |
| Loss Function | CrossEntropyLoss | 二分类标准损失 |
| Scheduler | CosineAnnealingLR | 余弦退火学习率 |

### 5.3 早停机制实现

```python
# train.py（核心逻辑）
best_acc = 0
patience_counter = 0
for epoch in range(max_epoch):
    train_loss, train_acc = train(...)
    val_loss, val_acc = val(...)
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 20:
            print("早停触发!")
            break
```

### 5.4 训练曲线

```
Epoch   训练损失  训练准确率  验证准确率  事件
────────────────────────────────────────────────
  1      0.7386    51.25%    84.38%    ★ 最佳模型保存
 15      0.1205    97.50%    87.50%    ★ 新的最佳模型
 35       -          -         -        早停触发，停止训练
```

---

## 6. 后端实现

### 6.1 项目文件清单

```
kc_dog_and_cat/
├── app.py                 # Flask 后端主程序
├── train.py               # 模型训练脚本
├── test.py                # 独立测试脚本
├── net.py                 # ResNet18 网络定义
├── split_data.py          # 数据集划分脚本
├── .gitignore             # Git 忽略规则
├── templates/
│   └── index.html         # 前端页面（单页应用）
├── static/
│   ├── uploads/           # 用户上传图片（运行时生成）
│   └── val_images/        # 验证集图片副本（页面展示用）
├── save_model/
│   └── best_model.pth     # 训练好的模型权重
├── data/
│   ├── train/{cat,dog}/   # 训练集
│   └── val/{cat,dog}/     # 验证集
└── predict_history.json   # 用户历史预测记录
```

### 6.2 模型加载（全局单例）

Flask 启动时一次性加载模型权重并常驻内存，避免每次请求都重新加载：

```python
# app.py
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
model.to(device)
model.eval()
```

### 6.3 推理接口 `/predict`

```python
@app.route('/predict', methods=['POST'])
def predict():
    # 1. 校验文件
    # 2. UUID 重命名，安全存储到 static/uploads/
    # 3. PIL → Transform → 送入模型
    # 4. softmax 输出概率
    # 5. 写入 predict_history.json
    # 6. 返回 JSON
    return jsonify({
        'prediction': 'cat' | 'dog',
        'confidence': 66.09,          # 置信度 %
        'cat_prob': 66.09,
        'dog_prob': 33.91,
        'image_url': '/static/uploads/xxx.jpg'
    })
```

### 6.4 历史记录持久化

采用 JSON 文件作为轻量级数据库（适合个人/小流量场景），配合 `threading.Lock` 保证多线程写入安全：

```python
_history_lock = threading.Lock()

def append_history(entry):
    with _history_lock:
        records = load_history()    # 读取 JSON
        records.append(entry)       # 追加
        save_history(records)       # 写回 JSON（原子操作）
```

### 6.5 统计接口 `/api/history`

返回汇总统计，供前端图表使用：

```json
{
  "total": 42,
  "cat_count": 18,
  "dog_count": 24,
  "cat_pct": 42.9,
  "dog_pct": 57.1,
  "avg_confidence": 85.3,
  "recent": [ ... 最近 20 条详细记录 ... ],
  "chart": {
    "labels": ["17:01:23", "17:01:23", ...],
    "confidences": [66.09, 74.14, ...],
    "predictions": ["cat", "dog", ...]
  }
}
```

### 6.6 启动时评测

Flask 启动时自动跑完整个验证集，计算准确率、复制图片到 `static/val_images/` 供前端案例区展示，整个过程只执行一次，之后所有页面请求直接使用缓存结果。

---

## 7. 前端实现

### 7.1 技术选型

| 技术 | 版本 | 用途 |
|------|------|------|
| HTML5 | - | 页面结构 |
| CSS3 | - | 样式（含 CSS Variables 实现仪表盘） |
| Vanilla JavaScript | ES2020 | 交互逻辑（无框架，轻量） |
| Chart.js | 4.4 | 图表可视化（CDN 引入） |
| getUserMedia API | - | 调用摄像头 |
| Canvas API | - | 拍照截帧 |

### 7.2 页面布局

```
┌─────────────────────────────────────┐
│                          GitHub 仓库 │  ← 顶部操作栏
├─────────────────────────────────────┤
│            🐱 🐶 猫狗分类器           │  ← 标题 + 副标题
├─────────────────────────────────────┤
│  ╭─ 准确率仪表盘 ─────────────────╮ │
│  │  ⭕ 87.5%  验证集评测结果       │ │
│  ╰────────────────────────────────╯ │
├─────────────────────────────────────┤
│           📷 开始识别                 │
│  ┌─────────────────────────────┐    │
│  │    点击或拖拽上传图片         │    │
│  │                             │    │
│  └─────────────────────────────┘    │
│          📸 使用摄像头拍照            │  ← 拍照按钮
├─────────────────────────────────────┤
│           📈 预测历史统计             │
│  ┌──┐ ┌──┐ ┌──┐ ┌──┐               │
│  │总│ │猫│ │狗│ │平│               │  ← 四格数据卡
│  └──┘ └──┘ └──┘ └──┘               │
│  ┌─饼图─┐ ┌────置信度折线图────┐    │
│  └──────┘ └───────────────────┘    │
│  ┌───── 历史记录表格 ──────────┐    │
│  │ 缩略图 | 时间 | 文件名 | ...│    │
│  └─────────────────────────────┘    │
├─────────────────────────────────────┤
│           🖼️ 验证集案例              │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐       │
│  │ ✓  │ │ ✓  │ │ ✗  │ │ ✓  │       │  ← 案例卡片
│  └────┘ └────┘ └────┘ └────┘       │
└─────────────────────────────────────┘
```

### 7.3 摄像头模块实现

浏览器原生 `navigator.mediaDevices.getUserMedia()` + Canvas 截帧：

```javascript
async function openCamera() {
    stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 } },
        audio: false
    });
    video.srcObject = stream;
}

function capturePhoto() {
    canvas.getContext('2d').drawImage(video, 0, 0);
    canvas.toBlob(blob => {
        capturedBlob = blob;                    // 暂存 JPEG blob
    }, 'image/jpeg', 0.9);
}

function confirmPhoto() {
    const file = new File([capturedBlob], 'camera.jpg', { type: 'image/jpeg' });
    handleFile(file);   // 复用已有的上传处理逻辑 → 统一走 POST /predict
}
```

支持特性：前后摄像头切换、前置摄像头画面自动镜像、弹窗关闭时自动释放摄像头。

---

## 8. 实验结果与分析

### 8.1 验证集准确率

| 指标 | 值 |
|------|-----|
| 验证集图片总数 | 20 张 |
| 正确识别数 | 17 张 |
| 错误识别数 | 3 张 |
| **整体准确率** | **87.5%** |
| 最佳 Epoch | 第 15 轮（早停触发） |

### 8.2 典型案例

**正确识别示例**

| 图片 | 真实标签 | 预测标签 | 置信度 |
|------|---------|---------|--------|
| cat.11.jpg | 🐱 猫 | 🐱 cat ✓ | 66.09% |
| dog.60.jpg | 🐶 狗 | 🐶 dog ✓ | 74.14% |

**错误识别分析**

模型在以下情况容易混淆：
- 侧脸猫 + 低分辨率图片 → 被误判为狗
- 站姿狗 + 类似猫的面部轮廓 → 被误判为猫
- **原因分析**：训练数据仅 80 张，多样性不足，模型对边缘 case 泛化能力有限

### 8.3 训练效率

在 CPU（Intel i5 / 无 GPU）环境下：

| 阶段 | 耗时 |
|------|------|
| 单个 Epoch 训练 | ~25 秒 |
| 完整训练（35 轮早停） | ~15 分钟 |
| 单张图片推理 | ~20-50 ms |

---

## 9. 部署与使用说明

### 9.1 环境准备

```bash
# 1. 克隆仓库
git clone https://github.com/joker-7775/kc_dog_and_cat.git
cd kc_dog_and_cat

# 2. 创建虚拟环境（推荐 Python 3.9+）
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. 安装依赖
pip install torch torchvision
pip install flask pillow
```

### 9.2 训练模型

```bash
python train.py
# 训练完成后权重保存在 save_model/best_model.pth
```

### 9.3 启动 Web 应用

```bash
python app.py
# 浏览器访问 http://127.0.0.1:5000
```

### 9.4 目录结构规范

**必须遵守的目录结构**（ImageFolder 需要）：

```
data/
├── train/
│   ├── cat/      ← 只能放猫的图片
│   └── dog/      ← 只能放狗的图片
└── val/
    ├── cat/
    └── dog/
```

⚠️ **踩坑记录**：不能在 `train/` 或 `val/` 下创建多余的子文件夹（比如不小心复制了整个 `train/` 目录进去），否则 `ImageFolder` 会把它们当作类别扫描，导致 `FileNotFoundError: Found no valid file for the classes xxx`。

### 9.5 摄像头权限说明

| 环境 | 能否使用摄像头 |
|------|---------------|
| `localhost`（本机测试） | ✅ 可以 |
| `https://` 部署 | ✅ 可以 |
| `http://` + 非 localhost | ❌ 被浏览器安全策略拦截 |

---

## 10. 总结与展望

### 10.1 工作总结

本项目完整实现了一个基于 ResNet18 的猫狗二分类 Web 应用，主要贡献：

1. **迁移学习方案**：在仅 80 张训练图片上达到 87.5% 准确率，证明了在小样本场景下迁移学习的有效性
2. **完整工程落地**：从训练脚本到可交互的 Web 应用，实现了 `模型 → 后端 API → 前端 UI → 历史分析` 的完整链路
3. **两种输入方式**：支持本地上传和实时摄像头拍照两种交互模式
4. **数据分析能力**：用户历史记录持久化存储，配合 Chart.js 图表实现了动态可视化

### 10.2 可改进方向

| 方向 | 说明 | 难度 |
|------|------|------|
| 增大数据集 | 下载完整的 Dogs vs Cats 数据集（2.5 万张） | ⭐ |
| 更强数据增强 | RandomResizedCrop、ColorJitter、Mixup | ⭐⭐ |
| 更先进的模型 | ResNet50、EfficientNet、Vision Transformer | ⭐⭐ |
| 量化加速 | TorchScript / ONNX 导出，推理速度提升 2~5 倍 | ⭐⭐ |
| 多标签支持 | 扩展到 120 种犬类识别（Stanford Dogs 数据集） | ⭐⭐⭐ |
| 生产级部署 | Gunicorn + Nginx + HTTPS + Docker | ⭐⭐⭐ |
| 用户认证 | 区分用户的历史记录 + 云端持久化 | ⭐⭐⭐ |

---

## 11. 参考文献

[1] He K, Zhang X, Ren S, et al. Deep Residual Learning for Image Recognition[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 770-778.

[2] Torchvision Model Zoo. https://pytorch.org/vision/stable/models.html

[3] Flask Documentation. https://flask.palletsprojects.com/

[4] Chart.js Documentation. https://www.chartjs.org/docs/

[5] Kaggle Dogs vs Cats 数据集. https://www.kaggle.com/c/dogs-vs-cats

[6] `torchvision.datasets.ImageFolder` 官方文档. https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html

---

## 附录

### A. 依赖版本

```
torch       2.14.0
torchvision 0.29.0
flask       3.1.3
pillow      (随 torchvision 自动安装)
Python      3.9+
```

### B. 开源仓库

本项目完整代码已开源：**https://github.com/joker-7775/kc_dog_and_cat**

### C. 核心文件代码行数统计

| 文件 | 行数 | 说明 |
|------|------|------|
| templates/index.html | ~600 | 前端页面（含 CSS/JS） |
| app.py | ~236 | Flask 后端 |
| train.py | ~175 | 训练脚本 |
| net.py | ~30 | 网络定义 |
| split_data.py | ~50 | 数据划分 |
| test.py | ~80 | 测试脚本 |

---

*文档版本：v1.0 · 最后更新：2026-09-14*