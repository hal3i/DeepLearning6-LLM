import unittest
import torch
from multihead_attention import MultiHeadAttention

class TestMultiHeadAttention(unittest.TestCase):
    def test_multihead_attention(self):
        embed_dim = 512
        n_head = 8
        head_dim = 64

        sut = MultiHeadAttention(embed_dim, n_head, head_dim)

        batch_size = 2
        context_len = 10
        x = torch.randn(batch_size, context_len, embed_dim)

        output = sut.forward(x)

        self.assertEqual("torch.Size([2, 10, 512])", f"{x.shape}")
        self.assertEqual("torch.Size([2, 10, 512])", f"{output.shape}")

if __name__ == '__main__':
    unittest.main()