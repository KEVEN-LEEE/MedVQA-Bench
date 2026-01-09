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
import matplotlib.pyplot as plt  # [新增] 引入绘图库

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
        self.batch_size = 128      # 4090显存充足，增大Batch Size以稳定BN层
        self.epochs = 100          # 实际上有了Early Stopping不需要太多
        
        # 学习率 - 分层微调策略
        self.lr_cnn = 1e-5         # CNN解冻，但使用极低学习率防止破坏预训练权重
        self.lr_lstm = 5e-4        # LSTM可以稍高
        self.lr_head = 1e-3        # 分类头/注意力层正常学习率
        self.weight_decay = 1e-4   # 稍微增加正则化力度
        
        # 模型参数
        self.embedding_dim = 300   # 增加词嵌入维度 (通常300比128效果好)
        self.hidden_dim = 512      # 增加LSTM隐藏层
        self.num_lstm_layers = 2
        
        self.save_dir = "./vqa_medical_attention_4090"
        self.use_amp = True
        self.patience = 15         # 早停耐心值
        self.min_occurrence = 2

config = Config()
os.makedirs(config.save_dir, exist_ok=True)

# ===================== 2. 数据处理 (保持稳健性) =====================

# 增强数据增强的强度，对抗过拟合
train_transforms = T.Compose([
    T.Resize((256, 256)),               # 先放大
    T.RandomCrop((224, 224)),           # 再随机裁剪
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=20),       # 增加旋转角度
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), # 增强色彩抖动
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
    
    # 答案词表
    ans_counts = Counter(all_answers)
    ans_vocab = sorted([ans for ans, cnt in ans_counts.items() if cnt >= min_occurrence])
    answer2idx = {ans: i for i, ans in enumerate(ans_vocab)}
    if "unk" not in answer2idx:
        answer2idx["unk"] = len(answer2idx)
    
    # 问题词表
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
            "labels": torch.tensor(label, dtype=torch.long)
        }

# ===================== 3. 模型定义 (引入注意力机制) =====================

class VisualAttention(nn.Module):
    """ 空间注意力机制：用问题向量去查询图像的每个区域 """
    def __init__(self, v_dim, q_dim, att_dim):
        super(VisualAttention, self).__init__()
        # 这里的 v_dim 是 CNN 通道数 (2048), q_dim 是 LSTM 输出 (hidden*2)
        self.v_proj = nn.Linear(v_dim, att_dim)
        self.q_proj = nn.Linear(q_dim, att_dim)
        self.alpha = nn.Linear(att_dim, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, v, q):
        """
        v: [batch, num_pixels, v_dim] -> [B, 49, 2048]
        q: [batch, q_dim]             -> [B, 512]
        """
        # 将问题向量复制以匹配图像像素数
        # q_rep: [B, 49, q_dim]
        q_rep = q.unsqueeze(1).expand(-1, v.size(1), -1)
        
        # 计算注意力分数
        # nonlinear(Wx*v + Wh*q + b)
        proj_v = self.v_proj(v)      # [B, 49, att_dim]
        proj_q = self.q_proj(q_rep)  # [B, 49, att_dim]
        
        h = torch.tanh(proj_v + proj_q)
        h = self.dropout(h)
        
        # 计算权重 alpha
        attn_scores = self.alpha(h)  # [B, 49, 1]
        attn_weights = F.softmax(attn_scores, dim=1) # [B, 49, 1]
        
        # 视觉特征加权求和
        # [B, 49, 1] * [B, 49, 2048] -> sum -> [B, 2048]
        v_weighted = (attn_weights * v).sum(dim=1)
        
        return v_weighted, attn_weights

class MedicalVQA_Attention(nn.Module):
    def __init__(self, num_classes, word_vocab_size):
        super().__init__()
        
        # 1. 图像编码器 (ResNet50)
        resnet = models.resnet50(pretrained=True)
        # 移除 avgpool 和 fc 层，保留空间特征 (B, 2048, 7, 7)
        modules = list(resnet.children())[:-2]
        self.cnn = nn.Sequential(*modules)
        self.cnn_dim = 2048
        
        # 2. 文本编码器 (LSTM)
        self.embedding = nn.Embedding(word_vocab_size, config.embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            config.embedding_dim, 
            config.hidden_dim, 
            config.num_lstm_layers, 
            bidirectional=True, 
            batch_first=True,
            dropout=0.3
        )
        self.lstm_dim = config.hidden_dim * 2
        
        # 3. 注意力层
        self.attention = VisualAttention(
            v_dim=self.cnn_dim,
            q_dim=self.lstm_dim,
            att_dim=512
        )
        
        # 4. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_dim + self.lstm_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, pixel_values, input_ids):
        # --- Image Branch ---
        # 注意：这里移除了 torch.no_grad()，允许微调 CNN
        # Out: [B, 2048, 7, 7]
        img_feat = self.cnn(pixel_values) 
        batch_size, channels, h, w = img_feat.size()
        # Reshape to [B, 49, 2048]
        img_feat = img_feat.view(batch_size, channels, -1).permute(0, 2, 1)
        
        # --- Text Branch ---
        embeds = self.embedding(input_ids)
        # LSTM output: [B, Seq, Dim], Hidden: [2*layers, B, Dim]
        self.lstm.flatten_parameters() # 优化内存
        lstm_out, (h_n, c_n) = self.lstm(embeds)
        
        # 获取句子级的特征 (取双向LSTM最后时刻的拼接)
        # h_n shape: [num_layers*num_directions, batch, hidden_size]
        # 取最后两层（双向）拼接
        q_feat = torch.cat([h_n[-2], h_n[-1]], dim=1) # [B, hidden*2]
        
        # --- Attention Fusion ---
        # 利用问题特征 q_feat 去关注图像特征 img_feat
        # v_att: [B, 2048]
        v_att, _ = self.attention(img_feat, q_feat)
        
        # --- Classification ---
        # 拼接 加权后的图像特征 和 问题特征
        combined = torch.cat([v_att, q_feat], dim=1)
        logits = self.classifier(combined)
        
        return logits

# ===================== 4. 训练与验证 =====================
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
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
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return total_loss / len(loader), correct / total

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc="Val", leave=False):
        pixel_values = batch["pixel_values"].to(config.device)
        input_ids = batch["input_ids"].to(config.device)
        labels = batch["labels"].to(config.device)
        
        with autocast(enabled=config.use_amp):
            logits = model(pixel_values, input_ids)
            loss = criterion(logits, labels)
            
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return total_loss / len(loader), correct / total

# ===================== 5. 主程序 =====================
def main():
    print(f"Device: {config.device} (RTX 4090 Optimization Enabled)")
    
    train_raw = load_annotations(config.train_anno_path)
    val_raw = load_annotations(config.val_anno_path)
    
    answer2idx, word2idx = build_vocab(train_raw, config.min_occurrence)
    num_classes = len(answer2idx)
    word_vocab_size = len(word2idx)
    
    train_ds = MedicalVQADataset(train_raw, train_transforms, answer2idx, word2idx)
    val_ds = MedicalVQADataset(val_raw, val_transforms, answer2idx, word2idx)
    
    # 4090 高并发加载
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, 
        num_workers=8, pin_memory=True, prefetch_factor=2, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, 
        num_workers=8, pin_memory=True
    )
    
    print(f"Train samples: {len(train_ds)} | Classes: {num_classes}")
    
    # 初始化带注意力的模型
    model = MedicalVQA_Attention(num_classes, word_vocab_size).to(config.device)
    
    # 参数分组与差异化学习率
    # 1. CNN 部分 (需要微调，但学习率要低)
    cnn_params = list(map(id, model.cnn.parameters()))
    # 2. 其他部分 (LSTM, Attention, Classifier) 使用较高学习率
    base_params = filter(lambda p: id(p) not in cnn_params, model.parameters())
    
    optimizer = optim.AdamW([
        {'params': model.cnn.parameters(), 'lr': config.lr_cnn}, # 1e-5
        {'params': base_params, 'lr': config.lr_head}            # 1e-3
    ], weight_decay=config.weight_decay)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler(enabled=config.use_amp)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    best_acc = 0.0
    patience_cnt = 0
    
    # [新增] 用于存储历史记录以便绘图
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print(f"{'Epoch':<6} | {'Tr Loss':<8} | {'Tr Acc':<8} | {'Val Loss':<8} | {'Val Acc':<8}")
    print("-" * 60)
    
    for epoch in range(config.epochs):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        v_loss, v_acc = validate(model, val_loader, criterion)
        
        # [新增] 记录数据
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        
        scheduler.step(v_acc) # 根据验证集准确率调整LR
        
        save_status = ""
        if v_acc > best_acc:
            best_acc = v_acc
            patience_cnt = 0
            torch.save(model.state_dict(), os.path.join(config.save_dir, "best_model_attn.pth"))
            save_status = "(*)"
        else:
            patience_cnt += 1
            
        print(f"{epoch+1:<6} | {t_loss:<8.4f} | {t_acc:<8.4f} | {v_loss:<8.4f} | {v_acc:<8.4f} {save_status}")
        
        if patience_cnt >= config.patience:
            print(f"Early stopping triggered. Best Val Acc: {best_acc:.4f}")
            break

    # ===================== 6. 结果绘制与保存 (新增) =====================
    print("Plotting results...")
    
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    # 设置画布大小：
    # 要求上下两张1280x720，即总宽度1280像素，总高度1440像素。
    # Matplotlib默认dpi=100，所以宽设为12.8英寸，高设为14.4英寸。
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12.8, 10.8), dpi=100)
    
    # --- 图1：训练结果 (Training Results) ---
    color_loss = 'tab:orange'
    color_acc = 'tab:blue'
    
    ax1.set_title(f'Training Results (Max Acc: {max(history["train_acc"]):.4f})', fontsize=16)
    ax1.set_xlabel('Epochs', fontsize=18)
    
    # 左轴绘制 Loss
    ax1.set_ylabel('Training Loss', color=color_loss, fontsize=18)
    ax1.plot(epochs_range, history['train_loss'], color=color_loss, linewidth=2, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 右轴绘制 Accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Training Accuracy', color=color_acc, fontsize=18)
    ax2.plot(epochs_range, history['train_acc'], color=color_acc, linewidth=2, label='Train Acc')
    ax2.tick_params(axis='y', labelcolor=color_acc)
    
    # --- 图2：验证结果 (Validation Results) ---
    color_val_loss = 'tab:orange'
    color_val_acc = 'tab:blue'
    
    ax3.set_title(f'Validation Results (Best Acc: {best_acc:.4f})', fontsize=18)
    ax3.set_xlabel('Epochs', fontsize=18)
    
    # 左轴绘制 Loss
    ax3.set_ylabel('Validation Loss', color=color_val_loss, fontsize=18)
    ax3.plot(epochs_range, history['val_loss'], color=color_val_loss, linewidth=2, label='Val Loss')
    ax3.tick_params(axis='y', labelcolor=color_val_loss)
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    # 右轴绘制 Accuracy
    ax4 = ax3.twinx()
    ax4.set_ylabel('Validation Accuracy', color=color_val_acc, fontsize=18)
    ax4.plot(epochs_range, history['val_acc'], color=color_val_acc, linewidth=2, label='Val Acc')
    ax4.tick_params(axis='y', labelcolor=color_val_acc)
    
    # 调整布局以防止标签重叠
    plt.tight_layout()
    
    # 保存图片
    plot_save_path = os.path.join(config.save_dir, "training_validation_results.png")
    plt.savefig(plot_save_path)
    print(f"Results plot saved to: {plot_save_path}")

if __name__ == "__main__":
    main()