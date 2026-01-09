# Medical VQA Model Implementation and Results Analysis
## Prerequisites
### Python Version
Python 3.8 or higher is recommended (compatible with PyTorch 1.10+ and transformers library requirements).

### Required Libraries
Install the necessary dependencies using the following command:
```bash
pip install torch torchvision transformers pillow numpy scikit-learn matplotlib tqdm json5
```
- `torch` & `torchvision`: Core deep learning framework for model building and training.
- `transformers`: For CLIP model and tokenizer (used in `vlmv3_draw.py`).
- `pillow`: Image processing (loading and transforming images).
- `numpy`: Numerical computations for data handling.
- `scikit-learn`: For accuracy calculation (in `cnn_lstmv2_draw.py`).
- `matplotlib`: Plotting training/validation results.
- `tqdm`: Progress bar for training and data loading.
- `json5`: Flexible JSON file parsing (handles various annotation formats).

## Code Overview
This repository contains two advanced Medical Visual Question Answering (VQA) models tailored for medical image-text understanding tasks. Both models are optimized for high-performance hardware (e.g., RTX 4090) with features like mixed-precision training, memory-efficient data loading, and early stopping.

### 1. `cnn_lstmv2_draw.py` - CNN-LSTM with Visual Attention
#### Core Architecture
- **Image Encoder**: ResNet50 (fine-tuned with low learning rate to preserve pre-trained medical image features).
- **Text Encoder**: Bidirectional LSTM with word embedding (captures sequential dependencies in medical questions).
- **Visual Attention Mechanism**: Fuses image and text features by having text queries focus on relevant image regions (enhances alignment between visual content and question semantics).
- **Classifier**: Fully connected layers with batch normalization and dropout (reduces overfitting for medical classification tasks).

#### Key Optimizations
- Hierarchical learning rates (low for CNN backbone, higher for LSTM/attention/classifier).
- Data augmentation (random cropping, flipping, rotation, color jitter) to improve generalization.
- Mixed-precision training (AMP) for faster computation and reduced memory usage.
- Early stopping (patience=15) to prevent overfitting on validation data.

### 2. `vlmv3_draw.py` - CLIP-Based Fine-Tuner
#### Core Architecture
- **Backbone**: Pre-trained CLIP (Contrastive Language-Image Pre-training) model (combines vision and text encoders pre-trained on large-scale data).
- **Fine-Tuning Strategy**: Freezes most CLIP layers, only unfreezes the last encoder layers and projection heads (balances feature preservation and task adaptation).
- **Classifier**: Lightweight sequential layers (faster inference than CNN-LSTM while maintaining accuracy).

#### Key Optimizations
- BF16 mixed-precision training (optimized for RTX 30/40 series GPUs, faster than FP16 with better stability).
- In-memory image caching (eliminates disk I/O bottlenecks during training).
- Pre-tokenization of text data (reduces redundant computation in training loops).
- Cosine annealing learning rate scheduler (improves convergence for small medical datasets).

## Result Visualization Insertion
### Insertion Locations
Insert the training/validation result images in the **"Result Analysis"** section below. Ensure the images are placed in the same directory as the generated Markdown file, or update the file paths accordingly.

### Image 1: Training & Validation Results from `cnn_lstmv2_draw.py`
<img width="1280" height="1080" alt="cnn_lstm" src="https://github.com/user-attachments/assets/2c01e0f0-4410-47f2-a947-216ae24f389d" />

#### Analysis
- **Training Metrics**: The training accuracy peaks at **98.84%** (Max Acc: 0.9884), and the training loss shows a steady downward trend (from ~3.0 to below 0.5). This indicates the model effectively learns patterns from the training data without catastrophic forgetting.
- **Validation Metrics**: The best validation accuracy is **82.04%** (Best Acc: 0.8204), with validation loss decreasing initially and stabilizing around 1.6-1.8.
- **Key Insight**: The gap between training and validation accuracy (~16%) suggests mild overfitting, which is expected given the model's complexity (attention + LSTM + CNN). The early stopping mechanism (patience=15) prevents further overfitting by stopping training once validation accuracy plateaus.

### Image 2: Training & Validation Results from `vlmv3_draw.py`
<img width="1280" height="1080" alt="training_validation_results" src="https://github.com/user-attachments/assets/4c64e0ce-5825-45a3-be65-4c86d9b9695d" />


#### Analysis
- **Training Metrics**: The training accuracy reaches ~90% (y-axis: 0.9), and training loss drops from ~3.0 to ~0.4. The smoother loss curve compared to the CNN-LSTM model reflects the stability of CLIP's pre-trained features.
- **Validation Metrics**: The validation accuracy peaks at ~80% (y-axis: 0.80), with validation loss stabilizing around 1.6-2.0.
- **Key Insight**: The CLIP-based model achieves comparable validation accuracy to the CNN-LSTM+Attention model but with faster training speed (due to pre-trained feature reuse and lightweight classifier). The smaller gap between training and validation accuracy (~10%) indicates better generalization, making it more suitable for scenarios with limited medical data.

## Summary
| Model | Core Component | Key Advantage | Best Validation Accuracy | Training Speed |
|-------|----------------|---------------|--------------------------|----------------|
| CNN-LSTM + Attention | Visual Attention Mechanism | Strong image-text alignment | 82.04% | Moderate |
| CLIP-Based Fine-Tuner | Pre-trained CLIP Backbone | Fast training & good generalization | ~80% | Fast (GPU-optimized) |

Both models are well-suited for medical VQA tasks (e.g., answering clinical questions about medical images from the Slake1.0 dataset). The CNN-LSTM+Attention model excels in alignment precision, while the CLIP-based model offers a better speed-accuracy tradeoff.
