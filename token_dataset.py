import torch
from torch.utils.data import Dataset, DataLoader

class TokenDataset(Dataset):
    def __init__(self, tokens, context_len):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.context_len = context_len

    def __len__(self):
        return len(self.tokens) - self.context_len
    
    def __getitem__(self, idx):
        x = self.tokens[idx:idx+self.context_len]
        y = self.tokens[idx+1:self.context_len+1]
        return x, y
