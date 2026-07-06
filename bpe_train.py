from collections import defaultdict
from tqdm import tqdm
from pretokenize import pretokenize

def count_pairs(ids, counts=None):
    if counts is None:
        counts = defaultdict(int)

    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

def merge(ids, pair, new_id):
    merged_ids = []
    i = 0

    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            merged_ids.append(new_id)
            i += 2
        else:
            merged_ids.append(ids[i])
            i += 1
    
    return merged_ids

def train_bpe(input_text, vocab_size, end_token="<|endoftext|>"):
    texts = input_text.split(end_token)
    ids_list = []
    for text in texts:
        for pretoken in pretokenize(text):
            ids_list.append(list(pretoken.encode("utf-8")))

    num_merges = vocab_size - 256 - 1
    merge_rules = {}

    for step in tqdm(range(num_merges), desc="Traning BPE"):
        counts = defaultdict(int)
        for ids in ids_list:
            counts = count_pairs(ids, counts)

        if not counts:
            break

        best_pair = max(counts, key=lambda pair: (counts[pair], pair[0], pair[1]))

        new_id = 256 + step
        merge_rules[best_pair] = new_id

        for i in range(len(ids_list)):
            ids_list[i] = merge(ids_list[i], best_pair, new_id)

    return merge_rules