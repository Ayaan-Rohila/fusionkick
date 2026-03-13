# FusionKick Model

The FusionKick architecture code (Temporal Fusion Transformer with asymmetric attention masking and hierarchical variable selection) will be released here upon paper acceptance.

## What will be included

- `fusionkick_model.py` — full PyTorch implementation
- `train.py` — training script with AdamW, cosine annealing, label smoothing
- `evaluate.py` — per-class F1, confusion matrix, attention weight extraction
- `deploy/` — TensorRT INT8 quantization and Jetson Orin inference pipeline
- Pretrained model checkpoint

## Why not now

The simulation dataset (in `../simulation/`) is the primary reproducibility contribution of this paper. The model architecture is described in full detail in Section 4 and can be reimplemented from that description. We are releasing the training code after final acceptance to allow for minor revisions to the experimental setup.

## In the meantime

To run the simulation and generate the dataset:

```bash
cd simulation
pip install -r ../requirements.txt
python generate_dataset.py --output ../data --seed 42
```

Expected output: ~61,000 windows across train/val/test splits, matching Table 3 in the paper.
