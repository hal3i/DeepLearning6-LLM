import unittest
import torch
from attention import Attention

class TestAttention(unittest.TestCase):
    def test_attention(self):
        sut = Attention(embed_dim=256, key_dim=64)
        x = torch.randn(2, 5, 256)
        y = sut.forward(x)

        self.assertEqual("torch.Size([2, 5, 256])", f"{x.shape}")
        self.assertEqual("torch.Size([2, 5, 256])", f"{y.shape}")

if __name__ == '__main__':
    unittest.main()