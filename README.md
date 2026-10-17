# SCN MAE starter

| File | What it is |
|---|---|
| `mae_encoder_classification_example.ipynb` | Walkthrough: load the encoder, get latents, run a linear probe, fine-tune with a classification head, evaluate per image and per sample, and plug in your own latent-space module |
| `mae_encoder.py` | Self-contained MAE ViT-B/L encoder (plain PyTorch), the `MAEClassifier` head, layer-wise lr decay, lr schedule, transforms and metrics |

## Setup (one local GPU)

```bash
pip install torch torchvision timm scikit-learn pandas matplotlib jupyter
jupyter lab   # open the notebook from this folder so `import mae_encoder` works
```

Tested with torch 2.6 and timm 1.0, Python 3.11.

## What you need (paths go in the config cell of the notebook)

* **Data**: a task folder such as `SCN_finetune/enhanced/ad_cls/` with `train/`, `val/` and `test/` in ImageFolder layout, plus `metadata_test.csv` for sample-level evaluation.
* **Checkpoints**:
  * `mae_pretrain_B_mask50_enhanced/checkpoint-799.pth` (ViT-B, 1.3 GB)
  * `mae_pretrain_L_mask50_enhanced/checkpoint-799.pth` (ViT-L, 4 GB)

  These are full MAE checkpoints. `mae_encoder.export_encoder_weights(src, dst)` writes a smaller encoder-only copy.

## Minimal usage

```python
import mae_encoder as me
enc = me.load_mae_encoder("checkpoint-799.pth", size="B")   # 224x224 input
tokens = enc(images)                     # (B, 197, 768): [CLS] + 14x14 patch tokens
model = me.MAEClassifier(enc, num_classes=4)                 # avg-pool -> LayerNorm -> Linear
logits = model(images)
z = model.forward_latent(images)         # (B, 768) latent fed to the head
```

Normalise images with the enhanced-SCN statistics, `me.ENHANCED_MEAN` and `me.ENHANCED_STD`, or just use `me.build_transform`.

The default hyperparameters match the lab's downstream runs (`run_all_downstream_merged.sbatch`): batch 32 × accumulation 8, base lr 1e-3, 100 epochs, 5 warm-up epochs, layer decay 0.75, weight decay 0.05, Mixup 0.8 / CutMix 1.0, label smoothing 0.1, and drop-path 0.1 (B) or 0.2 (L).
