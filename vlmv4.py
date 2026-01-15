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
import pandas as pd

# ===================== 0. Environment & Hardware Settings =====================
warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()

# ===================== 1. Configuration (Config) =====================
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root_dir = "/root/yolov11/datasets/data/Slake1.0"
        self.train_anno_path = os.path.join(self.root_dir, "train.json")
        self.val_anno_path = os.path.join(self.root_dir, "validate.json")
        self.img_base_dir = os.path.join(self.root_dir, "imgs")
        self.clip_model_name = "/root/autodl-tmp/ClipModel"
        
        self.batch_size = 64
        self.max_text_length = 77
        self.epochs = 50
        
        self.lr_backbone = 1e-6
        self.lr_head = 3e-4
        self.weight_decay = 1e-2
        
        self.save_dir = "./vqa_medical_fast"
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        self.patience = 8
        self.min_occurrence = 3   
        self.num_workers = 4
        self.cache_ram = True

config = Config()
os.makedirs(config.save_dir, exist_ok=True)

# ===================== 2. Data Processing & Augmentation =====================
train_transforms = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
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
        self.cached_images = {}
        
        print(f"Pre-tokenizing {len(samples)} texts...")
        self.encodings = tokenizer(
            [str(s["question"]) for s in samples],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=config.max_text_length
        )

        if self.cache:
            print(f"Caching {len(samples)} images to RAM (Accelerated Mode)...")
            for i, sample in enumerate(tqdm(samples)):
                img_path = os.path.join(config.img_base_dir, sample["img_name"])
                try:
                    img = Image.open(img_path).convert("RGB")
                    self.cached_images[i] = img
                except:
                    self.cached_images[i] = Image.new("RGB", (224, 224), "black")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.cache and idx in self.cached_images:
            image = self.cached_images[idx]
        else:
            img_path = os.path.join(config.img_base_dir, self.samples[idx]["img_name"])
            try:
                image = Image.open(img_path).convert("RGB")
            except:
                image = Image.new("RGB", (224, 224), "black")
        
        if self.transform:
            image = self.transform(image)
            
        input_ids = self.encodings["input_ids"][idx]
        attention_mask = self.encodings["attention_mask"][idx]
        ans = self.samples[idx]["answer"]
        label = self.answer2idx.get(ans, self.unk_idx)
        
        return {
            "pixel_values": image, 
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
            "index": idx # [New]
        }

# ===================== 3. Model Definition =====================
class MedicalCLIPFineTuner(nn.Module):
    def __init__(self, clip_model_name, num_classes):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        
        for param in self.clip.parameters():
            param.requires_grad = False
            
        for param in self.clip.vision_model.encoder.layers[-1:].parameters():
            param.requires_grad = True
        for param in self.clip.visual_projection.parameters():
            param.requires_grad = True
        for param in self.clip.text_model.encoder.layers[-1:].parameters():
            param.requires_grad = True
        for param in self.clip.text_projection.parameters():
            param.requires_grad = True

        embed_dim = self.clip.config.projection_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 512), 
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(512, num_classes)
        )
        
        # [New] Statistics container
        self.last_cosine_sim = 0.0

    def forward(self, pixel_values, input_ids, attention_mask):
        vision_out = self.clip.vision_model(pixel_values=pixel_values)
        text_out = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        
        img_embeds = self.clip.visual_projection(vision_out[1])
        txt_embeds = self.clip.text_projection(text_out[1])
        
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True)
        
        # [New] Calculate Image-Text Cosine Similarity (Dot product after normalization)
        if not self.training:
            self.last_cosine_sim = (img_embeds * txt_embeds).sum(dim=1).mean().item()
        
        logits = self.classifier(torch.cat([img_embeds, txt_embeds], dim=1))
        return logits

# ===================== 4. Training & Validation Workflow =====================
def get_grad_norm(model):
    # [New] Calculate L2 norm of gradients for unfrozen layers as a reference for update intensity
    total_norm = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    grad_norms = []
    
    for batch in tqdm(loader, desc="Train", leave=False):
        pixel_values = batch["pixel_values"].to(config.device, non_blocking=True)
        input_ids = batch["input_ids"].to(config.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(config.device, non_blocking=True)
        labels = batch["labels"].to(config.device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type="cuda", dtype=config.dtype):
            logits = model(pixel_values, input_ids, attention_mask)
            loss = criterion(logits, labels)
        
        loss.backward()
        
        # Track gradient norms
        grad_norms.append(get_grad_norm(model))
        
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
    return total_loss / len(loader), correct / total, avg_grad_norm

@torch.no_grad()
def validate(model, loader, criterion, raw_samples, idx2ans):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    # Statistics containers
    sim_stats = []
    error_samples = []
    
    for batch in tqdm(loader, desc="Val", leave=False):
        pixel_values = batch["pixel_values"].to(config.device, non_blocking=True)
        input_ids = batch["input_ids"].to(config.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(config.device, non_blocking=True)
        labels = batch["labels"].to(config.device, non_blocking=True)
        indices = batch["index"]
        
        with torch.amp.autocast(device_type="cuda", dtype=config.dtype):
            logits = model(pixel_values, input_ids, attention_mask)
            loss = criterion(logits, labels)
        
        sim_stats.append(model.last_cosine_sim)
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        if len(error_samples) < 5:
            mistakes = (preds != labels).nonzero(as_tuple=True)[0]
            for idx in mistakes:
                if len(error_samples) >= 5: break
                global_idx = indices[idx].item()
                raw = raw_samples[global_idx]
                pred_ans = idx2ans.get(preds[idx].item(), "unk")
                error_samples.append(f"Q: {raw['question']} | True: {raw['answer']} | Pred: {pred_ans}")
        
    return (total_loss / len(loader), correct / total, 
            np.mean(sim_stats), error_samples)

# ===================== 5. Main Program =====================
def main():
    print(f"Device: {config.device} | Type: {config.dtype}")
    start_train_time = time.time()
    
    train_raw = load_annotations(config.train_anno_path)
    val_raw = load_annotations(config.val_anno_path)
    
    answer2idx = build_vocab(train_raw, config.min_occurrence)
    idx2ans = {v: k for k, v in answer2idx.items()}
    num_classes = len(answer2idx)
    
    tokenizer = CLIPTokenizer.from_pretrained(config.clip_model_name)
    train_ds = MedicalVQADataset(train_raw, tokenizer, transform=train_transforms, answer2idx=answer2idx, cache=config.cache_ram)
    val_ds = MedicalVQADataset(val_raw, tokenizer, transform=val_transforms, answer2idx=answer2idx, cache=config.cache_ram)
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    
    model = MedicalCLIPFineTuner(config.clip_model_name, num_classes).to(config.device)
    
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if "clip" in n and p.requires_grad], 'lr': config.lr_backbone},
        {'params': [p for n, p in model.named_parameters() if "classifier" in n and p.requires_grad], 'lr': config.lr_head}
    ], weight_decay=config.weight_decay)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    best_acc = 0.0
    best_epoch = 0
    patience_cnt = 0
    
    # [New] Statistics containers
    metrics_data = []
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"{'Epoch':<6} | {'Tr Loss':<8} | {'Tr Acc':<8} | {'Val Loss':<8} | {'Val Acc':<8} | {'Time':<6}")
    print("-" * 70)
    
    for epoch in range(config.epochs):
        t0 = time.time()
        
        t_loss, t_acc, grad_norm = train_one_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc, cosine_sim, err_cases = validate(model, val_loader, criterion, val_raw, idx2ans)
        
        epoch_time = time.time() - t0
        
        # Record metrics
        gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
        metrics_data.append({
            "Epoch": epoch + 1,
            "Train Loss": round(t_loss, 4),
            "Train Acc": round(t_acc, 4),
            "Val Loss": round(v_loss, 4),
            "Val Acc": round(v_acc, 4),
            "Time (s)": round(epoch_time, 1),
            "GPU Mem (GB)": round(gpu_mem, 2),
            "Update L2 (Grad)": round(grad_norm, 4), # Represent update intensity via Gradient L2
            "Img-Txt CosSim": round(cosine_sim, 4),  # Image-Text alignment degree
            "Sample Errors": str(err_cases[:1])
        })

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        
        scheduler.step()
        
        save_status = ""
        if v_acc > best_acc:
            best_acc = v_acc
            best_epoch = epoch + 1
            patience_cnt = 0
            torch.save(model.state_dict(), os.path.join(config.save_dir, "best_model.pth"))
            save_status = "*"
        else:
            patience_cnt += 1
            
        print(f"{epoch+1:<6} | {t_loss:<8.4f} | {t_acc:<8.4f} | {v_loss:<8.4f} | {v_acc:<8.4f} | {epoch_time:.1f}s {save_status}")
        
        if patience_cnt >= config.patience:
            print("Early stopping triggered.")
            break

    print(f"Total time: {(time.time() - start_train_time):.2f}s")

    # ===================== 6. Plotting Results (Unchanged) =====================
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

    # ===================== 7. Table Generation (New) =====================
    print("\nGenerating Statistical Report...")
    df = pd.DataFrame(metrics_data)
    
    csv_path = os.path.join(config.save_dir, "training_metrics_clip.csv")
    df.to_csv(csv_path, index=False)
    
    summary_cols = ["Epoch", "Val Acc", "Update L2 (Grad)", "Img-Txt CosSim"]
    print("\n[Summary Table]")
    print(df[summary_cols].to_string(index=False))
    
    final_report = {
        "Model": "CLIP+FineTuning",
        "Best Val Acc": best_acc,
        "Best Epoch": best_epoch,
        "Total Training Time (s)": round(time.time() - start_train_time, 2),
        "Max GPU Memory (GB)": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "Best Weights": os.path.join(config.save_dir, "best_model.pth"),
        "Config": vars(config)
    }
    
    with open(os.path.join(config.save_dir, "final_report.json"), "w") as f:
        json.dump(final_report, f, indent=4, default=str)
    
    print(f"\nDetailed metrics saved to: {csv_path}")
    print(f"Final report saved to: {os.path.join(config.save_dir, 'final_report.json')}")

if __name__ == "__main__":
    main()