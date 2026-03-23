# FGCL4Rec: Frequency-guided Dual-view Graph Contrastive Learning for Sequential Service Recommendation

This repository contains the implementation of FGCL4Rec, a sequential service recommendation model that leverages frequency-guided dual-view graph learning and contrastive learning to enhance recommendation performance.

Important Note: In this repository, the term `item` in code variables, file names and script parameters refers to `service` in sequential service recommendation scenarios. We retain the original code naming conventions to ensure implementation stability and avoid modifying the underlying code logic.

## 1. Environment Setup

### Required Dependencies
- Python 3.8 or higher
- PyTorch 1.10 or higher
- NumPy 1.21 or higher

### Installation
```bash
pip install -r requirements.txt
```

## 2. Dataset Preparation

The dataset should be placed in the `data/` directory. We provide a preprocessed dataset `grocery.npy` containing:
- User-item interaction sequences 
- Total number of users
- Total number of items

### Generate Graph Matrices
FGCL4Rec requires two precomputed matrices:
- Item transition frequency matrix (captures next-item relationships)
- Item co-occurrence frequency matrix (captures co-occurrence relationships)

Generate these matrices by running:
```bash
python process.py
```
This will create two files in the `data/` directory:

- `item_nxt_frequency.npy`: Item transition frequency matrix
- `item_co_occur_frequency.npy`: Item co-occurrence frequency matrix

## 3. Pre-stored Model for Result Reproduction

For quickly verifying the model's performance and avoiding accidental overwriting of key checkpoints, we provide a pre-stored optimal model:  
- **Pre-stored model path**: `best_model_save.pth`

When evaluating, we recommend using this pre-stored model to replicate the reported performance. Note that this pre-stored model represents a **single optimal run** (with fixed random seed 3047) and may differ slightly from the results reported in the paper.  

## 4. Model Training

Train FGCL4Rec using the following command. By default, the model will save the best-performing checkpoint to `best_model.pth` (separate from the pre-stored model):

```bash
python main.py \
    --dataset data/grocery.npy \
    --item_sim data/item_co_occur_frequency.npy \
    --item_adj data/item_nxt_frequency.npy \
    --max_len 50 \
    --hidden_units 128 \
    --batch_size 64 \
    --lr 0.001 \
    --num_epochs 200 \
    --device cuda:0 \
    --save_path best_model.pth  # Default save path (won't overwrite pre-stored model)
```

## 5. One-click Model Evaluation

### Evaluate Pre-stored Model (Recommended)
To replicate the results in the paper, evaluate the pre-stored model:

```bash
python main.py \
    --dataset data/grocery.npy \
    --item_sim data/item_co_occur_frequency.npy \
    --item_adj data/item_nxt_frequency.npy \
    --device cuda:0 \
    --save_path best_model_save.pth \
    --eval_only
```

### Evaluate Newly Trained Model
To evaluate a model you trained yourself (saved in best_model.pth by default):
```bash
python main.py \
    --dataset data/grocery.npy \
    --item_sim data/item_co_occur_frequency.npy \
    --item_adj data/item_nxt_frequency.npy \
    --device cuda:0 \
    --eval_only  # Defaults to using best_model.pth
```

### Evaluation Output
The evaluation will output key metrics including:

- NDCG@5, NDCG@10 (Normalized Discounted Cumulative Gain)
- HR@5, HR@10 (Hit Ratio)
- MRR@5, MRR@10 (Mean Reciprocal Rank)


## 6. File Structure
```bash
FGCL4Rec/
├── data/
│ ├── grocery.npy # Preprocessed dataset
│ ├── item_nxt_frequency.npy # Transition frequency matrix
│ └── item_co_occur_frequency.npy # Co-occurrence matrix
├── best_model_save.pth # Pre-stored optimal model (for result reproduction)
├── main.py # Main training/evaluation script
├── model.py # FGCL4Rec model implementation
├── process.py # Script to generate graph matrices
├── utils.py # Utility functions (sampling, evaluation, etc.)
├── requirements.txt # List of required dependencies
└── README.md # Documentation for setup and usage
```
