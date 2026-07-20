import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from itertools import cycle
import re
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from ch02.gpt import GPT
from ch01.bpe_tokenizer import BPETokenizer
from codebot.utils import generate, get_device
from grpo_dataset import GRPODataset

device = get_device()
tokenizer_path = 'codebot/merge_rules.pkl'
sft_model_path = 'codebot/model_sft.pt'
grpo_model_save_path = 'codebot/model_grpo.pt'

learning_rate = 7e-6
max_iters = 500
n_update_per_generation = 2
eval_interval = 10
epsilon = 0.2
group_size = 8
batch_size = 32

tokenizer = BPETokenizer.load_from  (tokenizer_path)
model = GPT.load_from(sft_model_path, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

old_model = GPT.load_from(sft_model_path, device=device)
old_model.eval()

dataset = GRPODataset(tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
data_iter = cycle(dataloader)

def calculate_reward(ground_truth, response):
    try:
        matches = re.findall(r'(-?\d+)', response)
        if matches:
            predicted = int(matches[-1])
            return 1.0 if predicted == ground_truth else 0.0
        return 0.0
    except:
        return 0.0

def generate_group(model, tokenizer, prompts, gts, group_size):
    all_prompts = []
    all_responses = []
    all_advantages = []

    for prompt, gt in zip(prompts, gts):
        responses = []
        for _ in range(group_size):
            full_text = generate(model, tokenizer, prompt, temperature=1.0)
            response = full_text[len(prompt):]
            responses.append(response)

        rewards = torch.tensor([calculate_reward(gt, r) for r in responses])
        advantages = rewards - rewards.mean()

        for response, advantage in zip(responses, advantages):
            all_prompts.append(prompt)
            all_responses.append(response)
            all_advantages.append(advantage)

    return all_prompts, all_responses, torch.stack(all_advantages)

def compute_probs(model, ids):
    logits = model(ids)
    probs = F.softmax(logits[:, :-1, :], dim=-1)
    labels = ids[:, 1:]

    token_probs = torch.gather(
        probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)

    return token_probs

def grpo_loss(model, old_model, ids, mask, advantages, epsilon=0.2):
    probs = compute_probs(model, ids)
    with torch.no_grad():
        old_probs = compute_probs(old_model, ids)

    ratio = probs / (old_probs + 1e-8)
    advantages = advantages.unsqueeze(-1)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages

    mask = mask[:, 1:]
    token_objective = torch.min(unclipped, clipped) * mask

    n_samples = ids.size(0)
    return -token_objective.sum() / n_samples

accuracies = []
current_accuracy = 0.0
pbar = tqdm(range(max_iters))

for i in pbar:
    prompts, gts = next(data_iter)

    old_model.load_state_dict(model.state_dict())

    all_prompts, all_responses, all_advantages = generate_group(
        old_model, tokenizer, prompts, gts, group_size
    )

    ids, mask = dataset.get_batch(all_prompts, all_responses, device)
    all_advantages = all_advantages.to(device)

    for _ in range(n_update_per_generation):
        optimizer.zero_grad()
        loss = grpo_loss(model, old_model, ids, mask, all_advantages, epsilon)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    if i % eval_interval == 0:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for prompt, gt in dataset.data:
                response = generate(model, tokenizer, prompt, temperature=0)
                reward = calculate_reward(gt, response)
                correct += reward > 0
                total += 1
        
        model.train()
        current_accuracy = correct / total * 100
        accuracies.append(current_accuracy)

    pbar.set_postfix(
        {'loss': f'{loss.item():.4f}', 'acc': f'{current_accuracy:.1f}%'}
    )

model.save(grpo_model_save_path)