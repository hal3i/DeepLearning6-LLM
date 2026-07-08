import unittest
import torch
from gpt import GPT

class TestGPT(unittest.TestCase):
    def test_gpt_forward(self):
        vocab_size = 100
        max_context_len = 256
        embed_dim = 384
        n_head = 6
        n_layer = 6
        ff_dim = 4 * embed_dim
        dropout_rate = 0.1

        sut = GPT(vocab_size, max_context_len, embed_dim, n_head, n_layer, ff_dim, dropout_rate)

        dummy_input = torch.randint(0, vocab_size, (1, max_context_len))
        logits = sut.forward(dummy_input)

        self.assertEqual("torch.Size([1, 256, 100])", f"{logits.shape}")

if __name__ == '__main__':
    unittest.main()