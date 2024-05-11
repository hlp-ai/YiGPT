import torch

from yigpt.config import GPTConfig
from yigpt.gpt2 import GPT
from yigpt.inference import generate

if __name__ == "__main__":
    conf = GPTConfig()

    model = GPT(conf)
    print(model)

    x = torch.tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8]], dtype=torch.long)  # (B, T)
    logits, loss = model(x)
    print(logits.shape)

    y = torch.tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8]], dtype=torch.long)  # (B, T)
    logits, loss = model(x, y)
    print(logits.shape, loss.shape)

    pred = generate(model, x, 16, 1024)
    print(pred.shape)
    print(pred)
