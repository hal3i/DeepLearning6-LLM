import torch
from torch.utils.data import Dataset

class GRPODataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        for i in range(1, 10):
            for j in range(1, 10):
                prompt = f"### Instruction:\n{i}+{j}=\n\n### Response:\n"
                ground_truth = i + j
                self.data.append((prompt, ground_truth))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_batch(self, prompts, responses, device):
        all_ids = []
        all_masks = []

        for prompt, response in zip(prompts, responses):
            prompt_ids = self.tokenizer.encode(prompt)
            response_ids = self.tokenizer.encode(response)

            ids = prompt_ids + response_ids
            mask = [0] * len(prompt_ids) + [1] * len(response_ids)

            all_ids.append(ids)
            all_masks.append(mask)

        max_len = max(len(ids) for ids in all_ids)
        padded_ids = []
        padded_masks = []
        for ids, mask in zip(all_ids, all_masks):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [0] * pad_len)
            padded_masks.append(mask + [0] * pad_len)
        
        ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        mask = torch.tensor(padded_masks, dtype=torch.float, device=device)

        return ids, mask