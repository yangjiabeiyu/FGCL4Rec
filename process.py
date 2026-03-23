from collections import defaultdict
import numpy as np


def get_occurrence(user2item, item_num):
    """Calculate item co-occurrence frequencies"""
    item2cnt = defaultdict(int)
    item_similarity = np.zeros((item_num + 1, item_num + 1))
    for user in user2item:
        item_seq = user2item[user]
        for i in range(len(item_seq) - 2):
            item_i = int(item_seq[i])
            item2cnt[item_i] += 1
            item_set = set()
            for j in range(len(item_seq) - 2):
                item_j = int(item_seq[j])
                if i == j or item_j in item_set:
                    continue
                item_similarity[item_i, item_j] += 1
                item_set.add(item_j)
    # Normalize by item occurrence count
    for i in range(1, item_num + 1):
        if item2cnt[i] != 0:
            item_similarity[i, :] /= item2cnt[i]
    return item_similarity


def get_next_occurrence_train(user2item, item_num):
    """Calculate item transition frequencies (next-item occurrence)"""
    item2cnt = defaultdict(int)
    item2nxt_item = np.zeros((item_num + 1, item_num + 1))
    for seq in user2item.values():
        for i in range(max(len(seq) - 3, 0)):
            cur, nxt = int(seq[i]), int(seq[i + 1])
            item2cnt[cur] += 1
            item2nxt_item[nxt, cur] += 1
    # Normalize by transition count
    for i in range(1, item_num + 1):
        if item2cnt[i] != 0:
            item2nxt_item[:, i] /= item2cnt[i]
    return item2nxt_item


if __name__ == '__main__':
    dataset_path = "data/grocery.npy"
    data = np.load(dataset_path, allow_pickle=True)
    user2item, item_num = data[[0, 2]]

    # Generate and save transition matrix
    item_transition_matrix = get_next_occurrence_train(user2item, item_num)
    with open("data/item_nxt_frequency.npy", 'wb') as f:
        np.save(f, item_transition_matrix)

    # Generate and save co-occurrence matrix
    item_cooccurrence_matrix = get_occurrence(user2item, item_num)
    with open("data/item_co_occur_frequency.npy", 'wb') as f:
        np.save(f, item_cooccurrence_matrix)