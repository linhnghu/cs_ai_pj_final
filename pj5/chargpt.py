"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.


Modified version of Andrej Karpathy's "mingpt" repo found here: https://github.com/karpathy/minGPT
"""


import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from gpt_model import Character_GPT



class CharDataset(Dataset):
    def __init__(self, data):
        #Configure block size
        self.block_size = 10

        #Define possible characters, and create mapping from number to character
        chars = sorted(list(set(data)))
        self.index2char = { i:ch for i,ch in enumerate(chars) }
        self.char2index = { ch:i for i,ch in enumerate(chars) }
        vocab_size = len(chars)
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.char2index[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:],dtype=torch.long)
        return x, y



def train_single_iteration(model, batch, optimizer):
    x, y = batch
    optimizer.zero_grad(set_to_none=True)
    loss = model.get_loss(x, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def main():
    parser = argparse.ArgumentParser(description="Train a character-level GPT on a text file.")
    parser.add_argument("--input", type=Path, default=Path("input.txt"))
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--prompt", default="Pacman")
    parser.add_argument("--generate-tokens", type=int, default=500)
    args = parser.parse_args()
    if args.iterations < 1 or args.batch_size < 1 or args.generate_tokens < 0:
        parser.error("iterations and batch size must be positive and generate-tokens nonnegative")
    try:
        train_dataset = CharDataset(args.input.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as error:
        parser.error(str(error))
    if len(train_dataset.data) <= train_dataset.block_size:
        parser.error("the input text must contain more characters than the context window")
    if not args.prompt or any(ch not in train_dataset.char2index for ch in args.prompt):
        parser.error("the prompt must be nonempty and use only characters in the input text")

    sampler = torch.utils.data.RandomSampler(
        train_dataset, replacement=True, num_samples=args.iterations * args.batch_size
    )
    train_loader = DataLoader(
        train_dataset, sampler=sampler, batch_size=args.batch_size
    )
    model = Character_GPT(
        train_dataset.block_size, n_embd=100, n_layer=6, vocab_size=train_dataset.vocab_size
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00004)
    context = torch.tensor([[train_dataset.char2index[ch] for ch in args.prompt]])
    for step, batch in enumerate(train_loader, start=1):
        loss = train_single_iteration(model, batch, optimizer)
        if step == 1 or step % 100 == 0 or step == args.iterations:
            print(f"Iteration {step}: loss={loss:.4f}")
            generated = model.generate(context, args.generate_tokens)[0].tolist()
            print("".join(train_dataset.index2char[index] for index in generated))


if __name__ == "__main__":
    main()
