import os
import json
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPModel, CLIPTokenizer
import torchvision.transforms as T
import warnings
import matplotlib.pyplot as plt
import time

# ===================== 0. 环境与硬件设置 =====================
warnings.filterwarnings("ignore")

# 启用 TF32 加速 (RTX 30/40 系列必备)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# 开启 CUDNN Benchmark 以自动寻找最快算子
torch.backends.cudnn.benchmark = True

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()

# ===================== 1. 配置 (Config) =====================
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 路径设置 (请确保路径正确)
        self.root_dir = "/root/yolov11/datasets/data/Slake1.0"
        self.train_anno_path = os.path.join(self.root_dir, "train.json")
        self.val_anno_path = os.path.join(self.root_dir, "validate.json")
        self.img_base_dir = os.path.join(self.root_dir, "imgs")
        self.clip_model_name = "/root/autodl-tmp/ClipModel"
        
        # 训练参数
        self.batch_size = 64      # 平衡速度与显存利用率
        self.max_text_length = 77
        self.epochs = 50
        
        # 学习率
        self.lr_backbone = 1e-6   # 极低的学习率微调骨干
        self.lr_head = 3e-4       # 较高的学习率训练分类头
        self.weight_decay = 1e-2
        
        self.save_dir = "./vqa_medical_fast"
        
        # 混合精度: 4090 推荐使用 BF16 (无需 Scaler，速度快且稳定)
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        self.patience = 8
        self.min_occurrence = 3   
        
        # 数据加载优化
        self.num_workers = 4      # 这种规模的数据集，4个worker足够，太多反而增加开销
        self.cache_ram = True     # [核心优化] 将图片缓存到内存，消除磁盘I/O

config = Config()
os.makedirs(config.save_dir, exist_ok=True)

# ===================== 2. 数据处理与增强 =====================

# 简化后的增强，速度更快
train_transforms = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    # 去掉了耗时的 ColorJitter，如果为了极限速度可以注释掉 Rotation
    T.ToTensor(),
    T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                std=(0.26862954, 0.26130258, 0.27577711))
])

val_transforms = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                std=(0.26862954, 0.26130258, 0.27577711))
])

def load_annotations(anno_path):
    if not os.path.exists(anno_path):
        raise FileNotFoundError(f"Annotation file not found: {anno_path}")
    with open(anno_path, "r", encoding="utf-8") as f:
        try:
            content = json.load(f)
        except:
            f.seek(0)
            content = [json.loads(line) for line in f if line.strip()]
    
    # 过滤无效数据
    valid_samples = [item for item in content if "img_name" in item and "question" in item and "answer" in item]
    for item in valid_samples:
        item["answer"] = str(item["answer"]).strip().lower()
    return valid_samples

def build_vocab(samples, min_occurrence):
    from collections import Counter
    counts = Counter([s["answer"] for s in samples])
    vocab = sorted([ans for ans, cnt in counts.items() if cnt >= min_occurrence])
    answer2idx = {ans: i for i, ans in enumerate(vocab)}
    if "unk" not in answer2idx: answer2idx["unk"] = len(answer2idx)
    print(f"Vocab Size: {len(answer2idx)}")
    return answer2idx

class MedicalVQADataset(Dataset):
    def __init__(self, samples, tokenizer, transform=None, answer2idx=None, cache=False):
        self.samples = samples
        self.transform = transform
        self.answer2idx = answer2idx
        self.unk_idx = answer2idx["unk"]
        self.cache = cache
        self.cached_images = {} # 内存缓存字典
        
        # [核心优化] 预处理所有文本，避免训练时重复 Tokenize
        print(f"Pre-tokenizing {len(samples)} texts...")
        self.encodings = tokenizer(
            [str(s["question"]) for s in samples],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=config.max_text_length
        )

        # 如果开启缓存，且数据量不大，直接加载进内存
        if self.cache:
            print(f"Caching {len(samples)} images to RAM (Accelerated Mode)...")
            for i, sample in enumerate(tqdm(samples)):
                img_path = os.path.join(config.img_base_dir, sample["img_name"])
                try:
                    # 预加载原始 PIL 对象，Transform 还是在 getitem 里做（为了数据增强）
                    img = Image.open(img_path).convert("RGB")
                    self.cached_images[i] = img
                except:
                    self.cached_images[i] = Image.new("RGB", (224, 224), "black")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. 获取图片
        if self.cache and idx in self.cached_images:
            image = self.cached_images[idx]
        else:
            # 没缓存才去读盘
            img_path = os.path.join(config.img_base_dir, self.samples[idx]["img_name"])
            try:
                image = Image.open(img_path).convert("RGB")
            except:
                image = Image.new("RGB", (224, 224), "black")
        
        # 2. 应用 Transform
        if self.transform:
            image = self.transform(image)
            
        # 3. 获取预处理好的文本
        input_ids = self.encodings["input_ids"][idx]
        attention_mask = self.encodings["attention_mask"][idx]
        
        # 4. 获取 Label
        ans = self.samples[idx]["answer"]
        label = self.answer2idx.get(ans, self.unk_idx)
        
        return {
            "pixel_values": image, 
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long)
        }

# ===================== 3. 模型定义 =====================
class MedicalCLIPFineTuner(nn.Module):
    def __init__(self, clip_model_name, num_classes):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        
        # 冻结大部分参数
        for param in self.clip.parameters():
            param.requires_grad = False
            
        # 只解冻少量层，减少计算量，提高速度
        for param in self.clip.vision_model.encoder.layers[-1:].parameters(): # 只解冻最后1层 Vision
            param.requires_grad = True
        for param in self.clip.visual_projection.parameters():
            param.requires_grad = True
        for param in self.clip.text_model.encoder.layers[-1:].parameters():   # 只解冻最后1层 Text
            param.requires_grad = True
        for param in self.clip.text_projection.parameters():
            param.requires_grad = True

        embed_dim = self.clip.config.projection_dim
        
        # 稍微精简的 Head，速度更快
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 512), 
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        vision_out = self.clip.vision_model(pixel_values=pixel_values)
        text_out = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        
        img_embeds = self.clip.visual_projection(vision_out[1])
        txt_embeds = self.clip.text_projection(text_out[1])
        
        # 归一化
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True)
        
        logits = self.classifier(torch.cat([img_embeds, txt_embeds], dim=1))
        return logits

# ===================== 4. 训练与验证流程 =====================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    # 使用 BF16 上下文，无需 Scaler，无需 scaler.step()
    # set_to_none=True 稍微快一点点
    for batch in tqdm(loader, desc="Train", leave=False):
        # 移动数据到 GPU
        pixel_values = batch["pixel_values"].to(config.device, non_blocking=True)
        input_ids = batch["input_ids"].to(config.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(config.device, non_blocking=True)
        labels = batch["labels"].to(config.device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type="cuda", dtype=config.dtype):
            logits = model(pixel_values, input_ids, attention_mask)
            loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return total_loss / len(loader), correct / total

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    for batch in tqdm(loader, desc="Val", leave=False):
        pixel_values = batch["pixel_values"].to(config.device, non_blocking=True)
        input_ids = batch["input_ids"].to(config.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(config.device, non_blocking=True)
        labels = batch["labels"].to(config.device, non_blocking=True)
        
        with torch.amp.autocast(device_type="cuda", dtype=config.dtype):
            logits = model(pixel_values, input_ids, attention_mask)
            loss = criterion(logits, labels)
            
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return total_loss / len(loader), correct / total

# ===================== 5. 主程序 =====================
def main():
    print(f"Device: {config.device} | Type: {config.dtype}")
    
    # 1. 准备数据
    train_raw = load_annotations(config.train_anno_path)
    val_raw = load_annotations(config.val_anno_path)
    
    answer2idx = build_vocab(train_raw, config.min_occurrence)
    num_classes = len(answer2idx)
    
    tokenizer = CLIPTokenizer.from_pretrained(config.clip_model_name)
    
    # 初始化 Dataset (开启内存缓存 cache=True)
    train_ds = MedicalVQADataset(train_raw, tokenizer, transform=train_transforms, answer2idx=answer2idx, cache=config.cache_ram)
    val_ds = MedicalVQADataset(val_raw, tokenizer, transform=val_transforms, answer2idx=answer2idx, cache=config.cache_ram)
    
    # DataLoader: worker 不宜过多，因为数据已经在内存里了
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    
    # 2. 模型
    model = MedicalCLIPFineTuner(config.clip_model_name, num_classes).to(config.device)
    
    # 注意：这里我们取消了 torch.compile，因为它在小数据集上的预热时间可能超过收益
    # 如果数据集很大，可以在这里解开注释：
    # model = torch.compile(model) 

    # 3. 优化器
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if "clip" in n and p.requires_grad], 'lr': config.lr_backbone},
        {'params': [p for n, p in model.named_parameters() if "classifier" in n and p.requires_grad], 'lr': config.lr_head}
    ], weight_decay=config.weight_decay)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # 4. 训练循环
    best_acc = 0.0
    patience_cnt = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"{'Epoch':<6} | {'Tr Loss':<8} | {'Tr Acc':<8} | {'Val Loss':<8} | {'Val Acc':<8} | {'Time':<6}")
    print("-" * 70)
    
    start_total = time.time()
    
    for epoch in range(config.epochs):
        t0 = time.time()
        
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc = validate(model, val_loader, criterion)
        
        epoch_time = time.time() - t0
        
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        
        scheduler.step()
        
        save_status = ""
        if v_acc > best_acc:
            best_acc = v_acc
            patience_cnt = 0
            torch.save(model.state_dict(), os.path.join(config.save_dir, "best_model.pth"))
            save_status = "*"
        else:
            patience_cnt += 1
            
        print(f"{epoch+1:<6} | {t_loss:<8.4f} | {t_acc:<8.4f} | {v_loss:<8.4f} | {v_acc:<8.4f} | {epoch_time:.1f}s {save_status}")
        
        if patience_cnt >= config.patience:
            print("Early stopping triggered.")
            break

    print(f"Total time: {(time.time() - start_total):.2f}s")

    # ===================== 绘图 =====================
    epochs_range = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(epochs_range, history['train_loss'], label='Tr Loss', color='tab:orange')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(epochs_range, history['train_acc'], label='Tr Acc', color='tab:blue')
    ax2.set_ylabel('Accuracy')
    ax1.set_title('Training')
    
    ax3.plot(epochs_range, history['val_loss'], label='Val Loss', color='tab:orange')
    ax3.set_ylabel('Loss')
    ax3.legend(loc='upper left')
    ax4 = ax3.twinx()
    ax4.plot(epochs_range, history['val_acc'], label='Val Acc', color='tab:blue')
    ax4.set_ylabel('Accuracy')
    ax3.set_title('Validation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.save_dir, "results.png"))
    print("Done.")

if __name__ == "__main__":
    main()