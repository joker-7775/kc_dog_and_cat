import os
import uuid
import json
import shutil
import random
import threading
from datetime import datetime
import torch
import torchvision.models as models
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

VAL_IMAGE_DIR = os.path.join(BASE_DIR, 'static', 'val_images')
os.makedirs(VAL_IMAGE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, 'save_model', 'best_model.pth')
VAL_ROOT = os.path.join(BASE_DIR, 'data', 'val')
HISTORY_FILE = os.path.join(BASE_DIR, 'predict_history.json')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
CLASSES = ['cat', 'dog']

_history_lock = threading.Lock()


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def save_history(records):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def append_history(entry):
    with _history_lock:
        records = load_history()
        records.append(entry)
        save_history(records)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model = model.to(device)
model.eval()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_single(img_path):
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output[0], dim=0)
        pred_idx = torch.argmax(probs).item()
    return pred_idx, probs[0].item(), probs[1].item()


def build_stats():
    if not os.path.isdir(VAL_ROOT):
        return {'accuracy': 0, 'correct': 0, 'total': 0, 'samples': []}

    all_samples = []
    for label, cls in enumerate(CLASSES):
        cls_dir = os.path.join(VAL_ROOT, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            fpath = os.path.join(cls_dir, fname)
            if os.path.isfile(fpath):
                all_samples.append((fpath, label, cls, fname))

    results = []
    correct = 0
    for fpath, label, cls, fname in all_samples:
        pred_idx, cat_prob, dog_prob = predict_single(fpath)
        ok = (pred_idx == label)
        if ok:
            correct += 1
        dest = os.path.join(VAL_IMAGE_DIR, fname)
        if not os.path.exists(dest):
            shutil.copy2(fpath, dest)
        results.append({
            'filename': fname,
            'truth': cls,
            'prediction': CLASSES[pred_idx],
            'cat_prob': round(cat_prob * 100, 2),
            'dog_prob': round(dog_prob * 100, 2),
            'correct': ok,
            'url': f'/static/val_images/{fname}'
        })

    random.seed(42)
    correct_items = [r for r in results if r['correct']]
    wrong_items = [r for r in results if not r['correct']]
    pick_correct = random.sample(correct_items, min(3, len(correct_items)))
    pick_wrong = random.sample(wrong_items, min(2, len(wrong_items)))
    showcase = pick_correct + pick_wrong
    random.shuffle(showcase)

    return {
        'accuracy': round(correct / len(all_samples) * 100, 1) if all_samples else 0,
        'correct': correct,
        'total': len(all_samples),
        'samples': showcase
    }


STATS = build_stats()


@app.route('/')
def index():
    return render_template('index.html', stats=STATS)


@app.route('/api/stats')
def api_stats():
    return jsonify(STATS)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': '未接收到图片'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': f'不支持的文件类型，允许: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(save_path)

    try:
        img = Image.open(save_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output[0], dim=0)
            pred_idx = torch.argmax(probs).item()
            cat_prob = probs[0].item()
            dog_prob = probs[1].item()

        predicted = CLASSES[pred_idx]
        confidence = max(cat_prob, dog_prob) * 100

        append_history({
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'prediction': predicted,
            'confidence': round(confidence, 2),
            'cat_prob': round(cat_prob * 100, 2),
            'dog_prob': round(dog_prob * 100, 2),
            'filename': file.filename,
            'image_url': f'/static/uploads/{unique_name}'
        })

        return jsonify({
            'prediction': predicted,
            'confidence': round(confidence, 2),
            'cat_prob': round(cat_prob * 100, 2),
            'dog_prob': round(dog_prob * 100, 2),
            'image_url': f'/static/uploads/{unique_name}'
        })

    except Exception as e:
        return jsonify({'error': f'推理失败: {str(e)}'}), 500


@app.route('/api/history')
def api_history():
    records = load_history()
    total = len(records)
    cat_count = sum(1 for r in records if r['prediction'] == 'cat')
    dog_count = total - cat_count
    avg_conf = round(sum(r['confidence'] for r in records) / total, 2) if total else 0

    recent = records[-20:][::-1]
    chart_data = records[-30:]

    return jsonify({
        'total': total,
        'cat_count': cat_count,
        'dog_count': dog_count,
        'cat_pct': round(cat_count / total * 100, 1) if total else 0,
        'dog_pct': round(dog_count / total * 100, 1) if total else 0,
        'avg_confidence': avg_conf,
        'recent': recent,
        'chart': {
            'labels': [r['time'].split(' ')[1] for r in chart_data],
            'confidences': [r['confidence'] for r in chart_data],
            'predictions': [r['prediction'] for r in chart_data]
        }
    })


@app.route('/api/history/clear', methods=['POST'])
def api_history_clear():
    save_history([])
    return jsonify({'ok': True, 'msg': '历史记录已清空'})


if __name__ == '__main__':
    print(f"\n模型加载完成，使用设备: {device}")
    print(f"模型路径: {MODEL_PATH}")
    print("启动 Flask 服务器...\n")
    app.run(host='0.0.0.0', port=5000, debug=True)