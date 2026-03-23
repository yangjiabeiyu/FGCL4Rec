import time
import argparse
from model import FGCL4Rec
from utils import *


def set_global_seeds(seed=3047):
    """Set random seeds for all libraries to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_best_model(args, model_path):
    """One-click evaluation function to load the best model and evaluate its performance"""
    # Set global seeds for evaluation
    set_global_seeds()

    print(f"Evaluating FGCL4Rec model from: {model_path}")
    print(f"Evaluation Configuration: {args}")

    # Load dataset
    data_all = np.load(args.dataset, allow_pickle=True)
    user2item, user_num, item_num = data_all
    _, _, item_test = data_partition(user2item, args.max_len)

    # Load precomputed matrices
    item_transition_matrix = np.load(args.item_adj)
    item_cooccurrence_matrix = np.load(args.item_sim)

    # Initialize model
    model = FGCL4Rec(
        user_num,
        item_num,
        item_transition_matrix,
        item_cooccurrence_matrix,
        args
    ).to(args.device)

    # Load best model parameters
    try:
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Evaluate on test set
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        test_res = evaluate(model, item_test, item_num, args.neg_num)

    t1 = time.time() - t0
    print(f"\nEvaluation completed in {t1:.2f} seconds")
    print("Test Set Performance:")
    print(f"NDCG@5: {test_res[0, 0]:.4f}, NDCG@10: {test_res[1, 0]:.4f}")
    print(f"HR@5: {test_res[0, 1]:.4f}, HR@10: {test_res[1, 1]:.4f}")
    print(f"MRR@5: {test_res[0, 2]:.4f}, MRR@10: {test_res[1, 2]:.4f}")


parser = argparse.ArgumentParser()
# Dataset Configuration
parser.add_argument('--dataset', default='data/grocery.npy', type=str, help='Path to user-item interaction data')
parser.add_argument('--item_sim', default='data/item_co_occur_frequency.npy', type=str,
                    help='Path to item co-occurrence matrix')
parser.add_argument('--item_adj', default='data/item_nxt_frequency.npy', type=str,
                    help='Path to item transition matrix')

# Model Configuration
parser.add_argument('--max_len', default=50, type=int, help='Maximum length of user interaction sequence')
parser.add_argument('--hidden_units', default=128, type=int, help='Dimension of item embeddings')
parser.add_argument('--K_neg', default=100, type=int, help='Number of candidate negative samples')
parser.add_argument('--k_neg', default=50, type=int, help='Number of final negative samples')
parser.add_argument('--attn_aug_flag', default=0, type=int, help='Attention view augmentation strategy flag')
parser.add_argument('--trans_aug_flag', default=0, type=int, help='Transition view augmentation strategy flag')
parser.add_argument('--node_aug_portion_mask', default=0.7, type=float, help='Mask portion for weight augmentation')
parser.add_argument('--node_aug_portion_shuffle', default=0.7, type=float,
                    help='Shuffle portion for weight augmentation')
parser.add_argument('--node_aug_portion_drop', default=0.5, type=float, help='Drop portion for edge augmentation')
parser.add_argument('--node_aug_portion_add', default=0.7, type=float, help='Add portion for edge augmentation')
parser.add_argument('--temperature', default=1.0, type=float, help='Temperature parameter for contrastive learning')
parser.add_argument('--lambda_intra', default=0.8, type=float, help='Weight for intra-view contrastive loss')
parser.add_argument('--lambda_cross', default=0.8, type=float, help='Weight for cross-view contrastive loss')
parser.add_argument('--mask_std', default=0.8, type=float, help='Standard deviation for mask noise')
parser.add_argument('--K_add', default=500, type=int, help='Top-k candidates for edge addition')

# Training Configuration
parser.add_argument('--batch_size', default=64, type=int, help='Training batch size')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--num_epochs', default=200, type=int, help='Maximum training epochs')
parser.add_argument('--dropout_rate', default=0.80, type=float, help='Dropout rate')
parser.add_argument('--l2_emb', default=0.0001, type=float, help='L2 regularization weight')

# Evaluation Configuration
parser.add_argument('--neg_num', default=100, type=int, help='Number of negative samples for evaluation')
parser.add_argument('--device', default='cuda:0', type=str, help='Training/evaluation device (cpu/cuda)')
parser.add_argument('--early_stop_patience', default=10, type=int, help='Patience for early stopping')
parser.add_argument('--save_path', default='best_model.pth', type=str, help='Path to save/load best model parameters')
parser.add_argument('--eval_only', action='store_true', help='Only evaluate the best model without training')
parser.add_argument('--seed', default=3047, type=int, help='Random seed for reproducibility')
args = parser.parse_args()

if __name__ == '__main__':
    # Set global seeds first
    set_global_seeds(args.seed)

    # If eval_only flag is set, directly evaluate the best model
    if args.eval_only:
        evaluate_best_model(args, args.save_path)
        exit()

    # Otherwise, perform training
    print(f"FGCL4Rec Training Configuration: {args}")

    # Load dataset
    data_all = np.load(args.dataset, allow_pickle=True)
    user2item, user_num, item_num = data_all
    item_train, item_valid, item_test = data_partition(user2item, args.max_len)

    # Dataset statistics
    total_length = sum([len(seq) for seq in item_train.values()])
    print(f'Number of users: {user_num}, Number of items: {item_num}')
    print(f'Average sequence length: {total_length / user_num + 2:.2f}')

    # Initialize data sampler with fixed seed
    sampler = WarpSampler(
        item_train,
        user_num,
        item_num,
        batch_size=args.batch_size,
        max_len=args.max_len,
        n_workers=3,
        seed=args.seed  # Pass the fixed seed to sampler
    )

    # Load precomputed matrices
    item_transition_matrix = np.load(args.item_adj)
    item_cooccurrence_matrix = np.load(args.item_sim)

    model = FGCL4Rec(
        user_num,
        item_num,
        item_transition_matrix,
        item_cooccurrence_matrix,
        args
    ).to(args.device)

    # Parameter initialization
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

    # Training setup
    model.train()
    t0 = time.time()
    num_batch = len(item_train) // args.batch_size
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-6
    )

    # Early stopping and best model tracking
    best_val_ndcg10 = -1.0
    best_test_results = None
    early_stop_counter = 0

    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        for step in range(num_batch):
            # Get batch data
            item_seq, item_pos, item_neg = sampler.next_batch()
            item_seq, item_pos, item_neg = (
                np.array(item_seq),
                np.array(item_pos),
                np.array(item_neg)
            )

            # Forward pass
            pos_logits, neg_logits, aug_logits = model(
                item_seq,
                item_pos,
                item_neg,
                step
            )
            pos_labels = torch.ones(pos_logits.shape, device=args.device)
            neg_labels = torch.zeros(neg_logits.shape, device=args.device)

            # Calculate base loss
            optimizer.zero_grad()
            indices = np.where(item_pos != 0)  # Ignore padding
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            # Add L2 regularization
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)

            # Add contrastive learning losses
            if aug_logits:
                (pos_attn, neg_attn, pos_trans, neg_trans,
                 pos_cross, neg_cross) = aug_logits

                # Intra-view contrastive loss
                loss += args.lambda_intra * bce_criterion(pos_attn, torch.ones_like(pos_attn))
                loss += args.lambda_intra * bce_criterion(neg_attn, torch.zeros_like(neg_attn))
                loss += args.lambda_intra * bce_criterion(pos_trans, torch.ones_like(pos_trans))
                loss += args.lambda_intra * bce_criterion(neg_trans, torch.zeros_like(neg_trans))

                # Cross-view contrastive loss
                loss += args.lambda_cross * bce_criterion(pos_cross, torch.ones_like(pos_cross))
                loss += args.lambda_cross * bce_criterion(neg_cross, torch.zeros_like(neg_cross))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Evaluation phase (every 2 epochs)
        if epoch % 2 == 0:
            model.eval()
            t1 = time.time() - t0
            with torch.no_grad():
                # Evaluate on validation set
                val_res = evaluate(model, item_valid, item_num, args.neg_num)
                val_ndcg10 = val_res[1, 0]  # NDCG@10 (second row corresponds to k=10)

                # Evaluate on test set (only track when validation is best)
                test_res = evaluate(model, item_test, item_num, args.neg_num)

            # Print metrics
            print(f'Epoch {epoch}, Time elapsed: {t1:.2f}s')
            print(f'Validation - NDCG@5: {val_res[0, 0]:.4f}, NDCG@10: {val_res[1, 0]:.4f}')
            print(f'Test       - NDCG@5: {test_res[0, 0]:.4f}, NDCG@10: {test_res[1, 0]:.4f}\n')

            # Update best model based on validation NDCG@10
            if val_ndcg10 > best_val_ndcg10:
                best_val_ndcg10 = val_ndcg10
                best_test_results = test_res
                torch.save(model.state_dict(), args.save_path)
                early_stop_counter = 0  # Reset counter
                print(f'Updated best model (NDCG@10: {best_val_ndcg10:.4f}) saved to {args.save_path}')
            else:
                early_stop_counter += 1
                print(f'Early stopping counter: {early_stop_counter}/{args.early_stop_patience}')
                if early_stop_counter >= args.early_stop_patience:
                    print(f'Early stopping triggered at epoch {epoch}')
                    break

            t0 = time.time()

    # Print final results
    print("\nFinal Results (Test set metrics corresponding to best validation NDCG@10):")
    print(f'Test NDCG@5: {best_test_results[0, 0]:.4f}, NDCG@10: {best_test_results[1, 0]:.4f}')
    print(f'Test HR@5: {best_test_results[0, 1]:.4f}, HR@10: {best_test_results[1, 1]:.4f}')
    print(f'Test MRR@5: {best_test_results[0, 2]:.4f}, MRR@10: {best_test_results[1, 2]:.4f}')
    sampler.close()
