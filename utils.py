import torch
import random
import numpy as np
from multiprocessing import Process, Queue


def random_neq(l, r, s):
    """Sample a random number in [l, r) not in set s"""
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(item_train, user_num, item_num, batch_size, max_len, result_queue, seed):
    """Generate training batches using multiprocessing with fixed seed"""
    # Set random seeds for this process
    random.seed(seed)
    np.random.seed(seed)

    def sample():
        # Randomly select a user with valid sequence length
        user = np.random.randint(1, user_num + 1)
        while len(item_train[user]) <= 1:
            user = np.random.randint(1, user_num + 1)

        # Initialize sequences
        item_seq = np.zeros([max_len], dtype=np.int32)
        item_pos = np.zeros([max_len], dtype=np.int32)
        item_neg = np.zeros([max_len], dtype=np.int32)
        nxt = item_train[user][-1]
        idx = max_len - 1
        item_set = set(item_train[user])

        # Populate sequences
        for item in reversed(item_train[user][:-1]):
            item_seq[idx] = item
            item_pos[idx] = nxt
            if nxt != 0:
                item_neg[idx] = random_neq(1, item_num + 1, item_set)
            nxt = item
            idx -= 1
            if idx == -1:
                break
        return item_seq, item_pos, item_neg

    while True:
        one_batch = []
        for _ in range(batch_size):
            one_batch.append(sample())
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    """Multiprocessing sampler with fixed seeds for reproducibility"""

    def __init__(self, item_train, user_num, item_num, batch_size=64, max_len=10, n_workers=1, seed=3047):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []

        # Create workers with distinct seeds based on the main seed
        for i in range(n_workers):
            # Each worker gets a unique seed derived from the main seed
            process_seed = seed + i
            self.processors.append(
                Process(
                    target=sample_function,
                    args=(item_train, user_num, item_num, batch_size, max_len, self.result_queue, process_seed)
                )
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        """Get next training batch"""
        return self.result_queue.get()

    def close(self):
        """Terminate sampler processes"""
        for p in self.processors:
            p.terminate()
            p.join()


def data_partition(user2item, max_len):
    """Partition data into training, validation, and test sets"""
    item_train = {}
    item_seq_valid_list, item_idx_valid_list = [], []
    item_seq_test_list, item_idx_test_list = [], []

    for user in user2item:
        item_seq = user2item[user]
        seq_len = len(item_seq)

        if seq_len > 2:  # Ensure sufficient length for splitting
            # Training set: all but last 2 items
            item_train[user] = item_seq[:-2]

            # Validation set: predict penultimate item
            item_seq_valid = np.zeros([max_len], dtype=np.int32)
            valid_start = max(max_len - seq_len + 2, 0)
            valid_end = max(seq_len - max_len - 2, 0)
            item_seq_valid[valid_start:] = item_seq[valid_end: -2]
            item_seq_valid_list.append(item_seq_valid)
            item_idx_valid_list.append(item_seq[-2])

            # Test set: predict last item
            item_seq_test = np.zeros([max_len], dtype=np.int32)
            test_start = max(max_len - seq_len + 1, 0)
            test_end = max(seq_len - max_len - 1, 0)
            item_seq_test[test_start:] = item_seq[test_end: -1]
            item_seq_test_list.append(item_seq_test)
            item_idx_test_list.append(item_seq[-1])

    return [item_train, [item_seq_valid_list, item_idx_valid_list], [item_seq_test_list, item_idx_test_list]]


def evaluate(model, data_eval, item_num, neg_num, batch_size=1024):
    """Evaluate model performance (NDCG, HR, MRR) with fixed randomness"""
    num = len(data_eval[0])
    item_seq, item_idx = data_eval
    item_idx_new = []

    # Generate negative samples for evaluation with fixed random state
    for seq, idx in zip(item_seq, item_idx):
        neg_sample = set(seq)
        neg_sample.add(0)
        neg_sample.add(idx)
        item_sub_idx = [idx]  # Positive item at position 0

        # Add negative samples
        for _ in range(neg_num):
            t = np.random.randint(1, item_num + 1)
            while t in neg_sample:
                t = np.random.randint(1, item_num + 1)
            item_sub_idx.append(t)
        item_idx_new.append(item_sub_idx)

    item_seq, item_idx_new = np.array(item_seq), np.array(item_idx_new)
    num_batch = num // batch_size
    k_list = [5, 10]  # Evaluation metrics at k=5 and k=10
    res = np.zeros([2, 3])  # [NDCG, HR, MRR] for each k

    for i in range(num_batch + 1):
        # Get batch data
        l, r = i * batch_size, min(num, (i + 1) * batch_size)
        predictions = -model.predict(item_seq[l: r], item_idx_new[l: r])  # Negative for ascending sort
        pred = predictions.detach().cpu().numpy()
        rank = pred.argsort(axis=-1).argsort(axis=-1)[:, 0]  # Rank of positive item

        # Calculate metrics
        for idx, k in enumerate(k_list):
            rank_k = rank[rank < k]
            res[idx, 1] += len(rank_k)  # HR
            res[idx, 0] += sum(1 / np.log2(r + 2) for r in rank_k)  # NDCG
            res[idx, 2] += sum(1 / (r + 1) for r in rank_k)  # MRR

    return res / num  # Return metrics


def neg_sample_attn(item_attn_occurrence, neg_num_1=500, neg_num_2=50):
    """Sample negative examples for attention view contrastive learning with fixed randomness"""
    item_sim_topk = torch.argsort(item_attn_occurrence)[: neg_num_1].tolist()
    neg_samples = torch.tensor(random.sample(item_sim_topk, k=neg_num_2))
    return neg_samples


def weight_shuffle(item_trans, portion=0.2):
    """Shuffle portion of weights for transition view augmentation"""
    neighbor = torch.nonzero(item_trans).squeeze().tolist()
    if isinstance(neighbor, int):
        neighbor = [neighbor]
    num_neighbor = len(neighbor)
    shuffle_num = int(portion * num_neighbor)

    if shuffle_num > 1:
        shuffled_neighbor = random.sample(neighbor, k=shuffle_num)
        sort_neighbor = np.sort(shuffled_neighbor)
        item_trans[sort_neighbor] = item_trans[shuffled_neighbor]


def weight_mask(item_trans, dev, std=0.2, portion=0.2):
    """Add noise to portion of weights for transition view augmentation"""
    neighbor = torch.nonzero(item_trans).squeeze().tolist()
    if isinstance(neighbor, int):
        neighbor = [neighbor]
    num_neighbor = len(neighbor)
    mask_num = int(portion * num_neighbor)

    if mask_num > 0:
        mask_neighbor = random.sample(neighbor, k=mask_num)
        mask = torch.normal(0, std, [mask_num]).to(dev) * item_trans[mask_neighbor]
        item_trans[mask_neighbor] += mask
        item_trans /= (1 + torch.sum(mask))


def node_drop(item_trans, item_attn, portion=0.2):
    """Drop portion of edges for attention view augmentation"""
    item_attn[item_trans == 0] = -9e5
    neighbor = torch.nonzero(item_trans).squeeze().tolist()
    if isinstance(neighbor, int):
        neighbor = [neighbor]
    num_neighbor = len(neighbor)
    drop_num = int(portion * num_neighbor)

    if drop_num > 0:
        drop_neighbor = random.sample(neighbor, k=drop_num)
        item_attn[drop_neighbor] = -9e5


def node_add(item_trans, item_attn, item_attn_occurrence, portion=0.2, topk=200):
    """Add portion of edges for attention view augmentation"""
    idx_potential_neighbor = item_trans == 0
    potential_neighbor = torch.nonzero(idx_potential_neighbor).squeeze().tolist()

    if isinstance(potential_neighbor, int):
        potential_neighbor = [potential_neighbor]

    num_aug_neighbor = len(item_trans) - 1 - len(potential_neighbor)
    add_num = int(portion * num_aug_neighbor)

    if add_num > 0:
        item_sim_topk = torch.argsort(item_attn_occurrence)[-topk:].tolist()
        add_num = min(add_num, topk)
        add_neighbor = random.sample(item_sim_topk, k=add_num)
        idx_potential_neighbor[add_neighbor] = False
        item_attn[idx_potential_neighbor] = -9e5
    else:
        item_attn[idx_potential_neighbor] = -9e5
