import os
import json
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
import torchvision.models as models
import warnings
import matplotlib.pyplot as plt
import pandas as pd  # [新增] 引入 Pandas 处理表格
import time          # [新增] 引入 Time 统计耗时

# ===================== 0. 环境设置 =====================
warnings.filterwarnings("ignore")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 
set_seed()

# ===================== 1. 配置参数 =====================
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root_dir = "/root/yolov11/datasets/data/Slake1.0"
        self.train_anno_path = os.path.join(self.root_dir, "train.json")
        self.val_anno_path = os.path.join(self.root_dir, "validate.json")
        self.img_base_dir = os.path.join(self.root_dir, "imgs")
        
        # 4090 优化参数
        self.max_text_length = 32
        self.batch_size = 128
        self.epochs = 100
        
        # 学习率
        self.lr_cnn = 1e-5
        self.lr_lstm = 5e-4
        self.lr_head = 1e-3
        self.weight_decay = 1e-4
        
        # 模型参数
        self.embedding_dim = 300
        self.hidden_dim = 512
        self.num_lstm_layers = 2
        
        self.save_dir = "./vqa_medical_attention_4090"
        self.use_amp = True
        self.patience = 15
        self.min_occurrence = 2

config = Config()
os.makedirs(config.save_dir, exist_ok=True)

# ===================== 2. 数据处理 =====================
train_transforms = T.Compose([
    T.Resize((256, 256)),
    T.RandomCrop((224, 224)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=20),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_annotations(anno_path):
    if not os.path.exists(anno_path):
        raise FileNotFoundError(f"标注文件不存在：{anno_path}")
    with open(anno_path, "r", encoding="utf-8") as f:
        try:
            annotations = json.load(f)
        except:
            f.seek(0)
            annotations = [json.loads(line) for line in f if line.strip()]
    
    valid_samples = []
    required = ["img_name", "question", "answer"]
    for item in annotations:
        if all(k in item for k in required):
            item["answer"] = str(item["answer"]).strip().lower()
            valid_samples.append(item)
    return valid_samples

def build_vocab(samples, min_occurrence):
    from collections import Counter
    all_answers = [s["answer"] for s in samples]
    all_questions = [s["question"].split() for s in samples]
    all_words = [word.lower() for q in all_questions for word in q]
    
    ans_counts = Counter(all_answers)
    ans_vocab = sorted([ans for ans, cnt in ans_counts.items() if cnt >= min_occurrence])
    answer2idx = {ans: i for i, ans in enumerate(ans_vocab)}
    if "unk" not in answer2idx: answer2idx["unk"] = len(answer2idx)
    
    word_counts = Counter(all_words)
    word_vocab = ["<pad>", "<unk>"] + sorted([word for word, cnt in word_counts.items() if cnt >= min_occurrence])
    word2idx = {word: i for i, word in enumerate(word_vocab)}
    
    print(f"Answer vocab: {len(answer2idx)} | Question vocab: {len(word2idx)}")
    return answer2idx, word2idx

class MedicalVQADataset(Dataset):
    def __init__(self, samples, transform=None, answer2idx=None, word2idx=None):
        self.samples = samples
        self.transform = transform
        self.answer2idx = answer2idx
        self.word2idx = word2idx
        self.unk_ans_idx = answer2idx["unk"]
        self.unk_word_idx = word2idx["<unk>"]
        self.pad_idx = word2idx["<pad>"]

    def __len__(self):
        return len(self.samples)

    def _tokenize(self, text):
        words = text.lower().split()
        tokens = []
        for word in words[:config.max_text_length]:
            tokens.append(self.word2idx.get(word, self.unk_word_idx))
        if len(tokens) < config.max_text_length:
            tokens += [self.pad_idx] * (config.max_text_length - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(config.img_base_dir, sample["img_name"])
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224), "black")
            
        if self.transform:
            image = self.transform(image)
            
        text = str(sample["question"])
        input_ids = self._tokenize(text)
        ans = sample["answer"]
        label = self.answer2idx.get(ans, self.unk_ans_idx)
        
        return {
            "pixel_values": image,
            "input_ids": input_ids,
            "labels": torch.tensor(label, dtype=torch.long),
            "index": idx # [新增] 返回索引以便追踪错误案例
        }

# ===================== 3. 模型定义 =====================

class VisualAttention(nn.Module):
    def __init__(self, v_dim, q_dim, att_dim):
        super(VisualAttention, self).__init__()
        self.v_proj = nn.Linear(v_dim, att_dim)
        self.q_proj = nn.Linear(q_dim, att_dim)
        self.alpha = nn.Linear(att_dim, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, v, q):
        q_rep = q.unsqueeze(1).expand(-1, v.size(1), -1)
        proj_v = self.v_proj(v)
        proj_q = self.q_proj(q_rep)
        h = torch.tanh(proj_v + proj_q)
        h = self.dropout(h)
        attn_scores = self.alpha(h)
        attn_weights = F.softmax(attn_scores, dim=1)
        v_weighted = (attn_weights * v).sum(dim=1)
        return v_weighted, attn_weights

class MedicalVQA_Attention(nn.Module):
    def __init__(self, num_classes, word_vocab_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.cnn = nn.Sequential(*modules)
        self.cnn_dim = 2048
        
        self.embedding = nn.Embedding(word_vocab_size, config.embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            config.embedding_dim, config.hidden_dim, config.num_lstm_layers, 
            bidirectional=True, batch_first=True, dropout=0.3
        )
        self.lstm_dim = config.hidden_dim * 2
        
        self.attention = VisualAttention(self.cnn_dim, self.lstm_dim, 512)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_dim + self.lstm_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
        # [新增] 用于统计的临时容器
        self.stat_attn_weights_mean = 0.0
        self.stat_lstm_out_std = 0.0

    def forward(self, pixel_values, input_ids):
        img_feat = self.cnn(pixel_values) 
        batch_size, channels, h, w = img_feat.size()
        img_feat = img_feat.view(batch_size, channels, -1).permute(0, 2, 1)
        
        embeds = self.embedding(input_ids)
        self.lstm.flatten_parameters()
        lstm_out, (h_n, c_n) = self.lstm(embeds)
        
        # [新增统计] 记录 LSTM 输出的标准差（衡量特征丰富度）
        if not self.training:
            self.stat_lstm_out_std = lstm_out.std().item()

        q_feat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        v_att, attn_weights = self.attention(img_feat, q_feat)
        
        # [新增统计] 记录 Attention 权重的最大值均值（衡量聚焦程度）
        if not self.training:
            self.stat_attn_weights_mean = attn_weights.max(dim=1)[0].mean().item()
        
        combined = torch.cat([v_att, q_feat], dim=1)
        logits = self.classifier(combined)
        return logits

# ===================== 4. 训练与验证 =====================
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    start_time = time.time()
    
    for batch in tqdm(loader, desc="Train", leave=False):
        pixel_values = batch["pixel_values"].to(config.device)
        input_ids = batch["input_ids"].to(config.device)
        labels = batch["labels"].to(config.device)
        
        optimizer.zero_grad()
        with autocast(enabled=config.use_amp):
            logits = model(pixel_values, input_ids)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    cost_time = time.time() - start_time
    return total_loss / len(loader), correct / total, cost_time

@torch.no_grad()
def validate(model, loader, criterion, raw_samples, idx2ans):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    # 统计容器
    error_samples = []
    attn_stats = []
    lstm_stats = []
    
    for batch in tqdm(loader, desc="Val", leave=False):
        pixel_values = batch["pixel_values"].to(config.device)
        input_ids = batch["input_ids"].to(config.device)
        labels = batch["labels"].to(config.device)
        indices = batch["index"]
        
        with autocast(enabled=config.use_amp):
            logits = model(pixel_values, input_ids)
            loss = criterion(logits, labels)
            
        # 收集核心特征统计
        attn_stats.append(model.stat_attn_weights_mean)
        lstm_stats.append(model.stat_lstm_out_std)

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # 收集错误案例 (仅收集前5个以控制输出)
        if len(error_samples) < 5:
            mistakes = (preds != labels).nonzero(as_tuple=True)[0]
            for idx in mistakes:
                if len(error_samples) >= 5: break
                global_idx = indices[idx].item()
                raw = raw_samples[global_idx]
                pred_ans = idx2ans.get(preds[idx].item(), "unk")
                error_samples.append(f"Q: {raw['question']} | True: {raw['answer']} | Pred: {pred_ans}")

    return (total_loss / len(loader), correct / total, 
            np.mean(attn_stats), np.mean(lstm_stats), error_samples)

# ===================== 5. 主程序 =====================
def main():
    start_train_time = time.time()
    print(f"Device: {config.device} (RTX 4090 Optimization Enabled)")
    
    train_raw = load_annotations(config.train_anno_path)
    val_raw = load_annotations(config.val_anno_path)
    
    answer2idx, word2idx = build_vocab(train_raw, config.min_occurrence)
    idx2ans = {v: k for k, v in answer2idx.items()}
    num_classes = len(answer2idx)
    word_vocab_size = len(word2idx)
    
    train_ds = MedicalVQADataset(train_raw, train_transforms, answer2idx, word2idx)
    val_ds = MedicalVQADataset(val_raw, val_transforms, answer2idx, word2idx)
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    model = MedicalVQA_Attention(num_classes, word_vocab_size).to(config.device)
    
    cnn_params = list(map(id, model.cnn.parameters()))
    base_params = filter(lambda p: id(p) not in cnn_params, model.parameters())
    optimizer = optim.AdamW([
        {'params': model.cnn.parameters(), 'lr': config.lr_cnn},
        {'params': base_params, 'lr': config.lr_head}
    ], weight_decay=config.weight_decay)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler(enabled=config.use_amp)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    best_acc = 0.0
    best_epoch = 0
    patience_cnt = 0
    
    # [新增] 详细统计数据容器
    metrics_data = []
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"{'Epoch':<6} | {'Tr Loss':<8} | {'Tr Acc':<8} | {'Val Loss':<8} | {'Val Acc':<8}")
    print("-" * 60)
    
    for epoch in range(config.epochs):
        t_loss, t_acc, t_time = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        v_loss, v_acc, att_mean, lstm_std, err_cases = validate(model, val_loader, criterion, val_raw, idx2ans)
        
        # 记录统计数据
        gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
        metrics_data.append({
            "Epoch": epoch + 1,
            "Train Loss": round(t_loss, 4),
            "Train Acc": round(t_acc, 4),
            "Val Loss": round(v_loss, 4),
            "Val Acc": round(v_acc, 4),
            "Time (s)": round(t_time, 1),
            "GPU Mem (GB)": round(gpu_mem, 2),
            "Attn Focus Score": round(att_mean, 4), # 注意力集中度
            "LSTM Activations": round(lstm_std, 4), # LSTM 活跃度
            "Sample Errors": str(err_cases[:1]) # 只留一个例子防爆表
        })
        
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        
        scheduler.step(v_acc)
        
        save_status = ""
        if v_acc > best_acc:
            best_acc = v_acc
            best_epoch = epoch + 1
            patience_cnt = 0
            torch.save(model.state_dict(), os.path.join(config.save_dir, "best_model_attn.pth"))
            save_status = "(*)"
        else:
            patience_cnt += 1
            
        print(f"{epoch+1:<6} | {t_loss:<8.4f} | {t_acc:<8.4f} | {v_loss:<8.4f} | {v_acc:<8.4f} {save_status}")
        
        if patience_cnt >= config.patience:
            print(f"Early stopping triggered. Best Val Acc: {best_acc:.4f}")
            break

    # ===================== 6. 结果绘制 (保持不变) =====================
    print("Plotting results...")
    epochs_range = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12.8, 10.8), dpi=100)
    
    color_loss = 'tab:orange'
    color_acc = 'tab:blue'
    
    ax1.set_title(f'Training Results (Max Acc: {max(history["train_acc"]):.4f})', fontsize=16)
    ax1.set_xlabel('Epochs', fontsize=18)
    ax1.set_ylabel('Training Loss', color=color_loss, fontsize=18)
    ax1.plot(epochs_range, history['train_loss'], color=color_loss, linewidth=2, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Training Accuracy', color=color_acc, fontsize=18)
    ax2.plot(epochs_range, history['train_acc'], color=color_acc, linewidth=2, label='Train Acc')
    ax2.tick_params(axis='y', labelcolor=color_acc)
    
    color_val_loss = 'tab:orange'
    color_val_acc = 'tab:blue'
    ax3.set_title(f'Validation Results (Best Acc: {best_acc:.4f})', fontsize=18)
    ax3.set_xlabel('Epochs', fontsize=18)
    ax3.set_ylabel('Validation Loss', color=color_val_loss, fontsize=18)
    ax3.plot(epochs_range, history['val_loss'], color=color_val_loss, linewidth=2, label='Val Loss')
    ax3.tick_params(axis='y', labelcolor=color_val_loss)
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    ax4 = ax3.twinx()
    ax4.set_ylabel('Validation Accuracy', color=color_val_acc, fontsize=18)
    ax4.plot(epochs_range, history['val_acc'], color=color_val_acc, linewidth=2, label='Val Acc')
    ax4.tick_params(axis='y', labelcolor=color_val_acc)
    
    plt.tight_layout()
    plot_save_path = os.path.join(config.save_dir, "training_validation_results.png")
    plt.savefig(plot_save_path)
    print(f"Results plot saved to: {plot_save_path}")

    # ===================== 7. 表格生成 (新增) =====================
    print("\nGenerating Statistical Report...")
    df = pd.DataFrame(metrics_data)
    
    # 保存详细 CSV
    csv_path = os.path.join(config.save_dir, "training_metrics_cnn_lstm.csv")
    df.to_csv(csv_path, index=False)
    
    # 打印概览 (不包含错误样本列以保持整洁)
    summary_cols = ["Epoch", "Train Loss", "Val Loss", "Val Acc", "Time (s)", "Attn Focus Score"]
    print("\n[Summary Table]")
    print(df[summary_cols].to_string(index=False))
    
    # 保存最终配置和最佳结果
    final_report = {
        "Model": "CNN+LSTM+Attention",
        "Best Val Acc": best_acc,
        "Best Epoch": best_epoch,
        "Total Training Time (s)": round(time.time() - start_train_time, 2),
        "Max GPU Memory (GB)": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "Best Weights": os.path.join(config.save_dir, "best_model_attn.pth"),
        "Config": vars(config)
    }
    
    with open(os.path.join(config.save_dir, "final_report.json"), "w") as f:
        json.dump(final_report, f, indent=4, default=str)
    
    print(f"\nDetailed metrics saved to: {csv_path}")
    print(f"Final report saved to: {os.path.join(config.save_dir, 'final_report.json')}")

if __name__ == "__main__":
    main()