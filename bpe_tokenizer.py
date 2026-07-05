from bpe_train import merge

class BPETokenizer:
    def __init__(self, merge_rules):
        self.merge_rules = merge_rules

        self.id_to_bytes = {i: bytes([i]) for i in range(256)}

        for (id1, id2), new_id in merge_rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]

        self.vocab_size = len(self.id_to_bytes)

    def encode(self, text):
        ids = list(text.encode("utf-8"))

        for merge_pair, new_id in self.merge_rules.items():
            ids = merge(ids, merge_pair, new_id)

        return ids
    
    def decode(self, ids):
        byte_list = [self.id_to_bytes[i] for i in ids]

        text_bytes = b"".join(byte_list)

        text = text_bytes.decode("utf-8", errors="replace")
        return text