"""
What this does:
- trains a HuggingFace GPT2 decoder-only model on the dataset
"""

"""
before running this, run:

!pip install transformers

from google.colab import drive
drive.mount("/content/drive", force_remount=True)
"""

"""
See LLM note at the top of tokenize_data.py
"""

import math
import os
import shutil
import time
import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

#config
prepared_data_directory = "/content/drive/MyDrive/MSAI/spring_2026_quarter/GenAI/text_generation/dataset1/prepared_data"
checkpoint_directory = "/content/drive/MyDrive/MSAI/spring_2026_quarter/GenAI/text_generation/dataset1/checkpoints"
local_prepared_data_directory = "/tmp/prepared"

#these must match the other files
vocab_size = 8000
block_size = 512

#model size hyperparameters
#starting with 8, 8, 512, 0.1
number_of_layers = 8
number_of_heads = 8
embedding_dimension = 512
dropout_rate = 0.1

#training
batch_size = 32
number_of_epochs = 10
learning_rate = 3e-4
gradient_clip = 1.0
warmup_steps = 200
print_progress_interval = 10_000
seed = 33


#this makes (input_ids, labels) pairs. Labels is -100 in the abstract and padding where we don't want loss
class BinDataset:
    def __init__(self, tokens_path, mask_path, block_size):
        tokens_mm = np.memmap(tokens_path, dtype=np.uint16, mode="r")
        mask_mm = np.memmap(mask_path, dtype=np.uint8, mode="r")
        self.n = tokens_mm.size // block_size
        self.tokens = tokens_mm.reshape(self.n, block_size)
        self.mask = mask_mm.reshape(self.n, block_size)
        self.block_size = block_size

    def __len__(self):
        return self.n

    def get_batch(self, indices):
        batch_tokens = np.asarray(self.tokens[indices], dtype=np.int64)
        batch_mask = np.asarray(self.mask[indices], dtype=np.int64)

        input_ids = torch.from_numpy(batch_tokens)
        labels = input_ids.clone()

        shifted_mask = np.zeros_like(batch_mask)
        shifted_mask[:, 1:] = batch_mask[:, :-1]
        labels[torch.from_numpy(shifted_mask) == 0] = -100

        return input_ids, labels

#this copies bin and tokenizer files from Drive to /tmp so it can be read faster
def stage_prep_locally():
    os.makedirs(local_prepared_data_directory, exist_ok=True)
    files_to_transfer = [
        "vocab.json", "merges.txt",
        "train_tokens.bin", "train_mask.bin",
        "val_tokens.bin", "val_mask.bin",
    ]
    for name in files_to_transfer:
        src = f"{prepared_data_directory}/{name}"
        dst = f"{local_prepared_data_directory}/{name}"
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            continue
        shutil.copy(src, dst)
        print(f"{name} copied to {dst}")
    return local_prepared_data_directory

#checkpointing
def find_checkpoints(checkpoint_directory):
    names = sorted(f for f in os.listdir(checkpoint_directory) if f.endswith(".pt"))
    return [f"{checkpoint_directory}/{name}" for name in names]

def save_checkpoint(model, optimizer, epoch, step, examples_seen, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = f"{ckpt_dir}/ckpt_epoch_{epoch:03d}.pt"
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "examples_seen": examples_seen,
        "config": model.config.to_dict(),
    }, tmp)
    os.replace(tmp, path)
    return path

#for periodically evaluating against val set
@torch.no_grad()
def evaluate(model, val_ds, batch_size, device):
    model.eval()
    n = len(val_ds)
    total_loss = 0.0
    total_count = 0
    indices = np.arange(n)
    for start in range(0, n, batch_size):
        index = indices[start:start + batch_size]
        input_ids, labels = val_ds.get_batch(index)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(input_ids=input_ids, labels=labels)
        n_tokens_contributing_to_loss = (labels != -100).sum().item()
        total_loss += output.loss.item() * n_tokens_contributing_to_loss
        total_count += n_tokens_contributing_to_loss
    model.train()
    return total_loss / max(total_count, 1)


#main
def main():
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda"

    local = stage_prep_locally()
    train_ds = BinDataset(f"{local}/train_tokens.bin", f"{local}/train_mask.bin", block_size)
    val_ds = BinDataset(f"{local}/val_tokens.bin", f"{local}/val_mask.bin", block_size)
    print(f"Train dataset: {len(train_ds)} examples. Validation dataset: {len(val_ds)} examples")

    #HuggingFace GPT2 model
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=block_size,
        n_ctx=block_size,
        n_embd=embedding_dimension,
        n_layer=number_of_layers,
        n_head=number_of_heads,
        resid_pdrop=dropout_rate,
        embd_pdrop=dropout_rate,
        attn_pdrop=dropout_rate,
    )
    model = GPT2LMHeadModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(f"Training for {number_of_epochs} epochs (batch size {batch_size})")

    start_epoch = 1
    global_step = 0
    examples_seen = 0
    checkpoints = find_checkpoints(checkpoint_directory)
    if checkpoints: #resume from latest checkpoint if there is 1
        latest_checkpoint_path = checkpoints[-1]
        print(f"Resuming from {os.path.basename(latest_checkpoint_path)}")
        latest_checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        model.load_state_dict(latest_checkpoint["model"])
        optimizer.load_state_dict(latest_checkpoint["optimizer"])
        start_epoch = latest_checkpoint["epoch"] + 1
        global_step = latest_checkpoint["step"]
        examples_seen = latest_checkpoint["examples_seen"]
        print(f"Resumed from checkpoint. epoch={start_epoch}, step={global_step}, examples_seen={examples_seen}")
        if start_epoch > number_of_epochs:
            print("All epochs already complete. Not doing any new training.")
            return

    #training loop
    next_progress = ((examples_seen // print_progress_interval) + 1) * print_progress_interval

    for epoch in range(start_epoch, number_of_epochs + 1):
        print()
        print(f"Epoch {epoch}/{number_of_epochs}")
        epoch_starting_time = time.time()
        model.train()
        indices = np.random.permutation(len(train_ds))
        total_loss_this_epoch = 0.0
        title_tokens_this_epoch = 0

        for i in range(0, len(train_ds), batch_size):
            batch_index = indices[i:i + batch_size]
            input_ids, labels = train_ds.get_batch(batch_index)
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            #learning rate increases over the warmup steps
            if global_step < warmup_steps:
                lr = learning_rate * (global_step + 1) / (warmup_steps + 1)
            else:
                lr = learning_rate

            for group in optimizer.param_groups:
                group["lr"] = lr

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16): #training is much faster with this dtype
                output = model(input_ids=input_ids, labels=labels)
                loss = output.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip) #gradient clipping
            optimizer.step()

            n_tokens_contributing_to_loss = (labels != -100).sum().item()
            total_loss_this_epoch += loss.item() * n_tokens_contributing_to_loss
            title_tokens_this_epoch += n_tokens_contributing_to_loss
            global_step += 1
            examples_seen += input_ids.size(0)

            if examples_seen >= next_progress:
                average_loss = total_loss_this_epoch / title_tokens_this_epoch
                print(f"step {global_step},  examples {examples_seen}, loss {average_loss},  lr {lr}")
                next_progress = ((examples_seen // print_progress_interval) + 1) * print_progress_interval

        train_avg = total_loss_this_epoch / title_tokens_this_epoch
        training_time = time.time() - epoch_starting_time

        #Validate
        validation_starting_time = time.time()
        val_loss = evaluate(model, val_ds, batch_size, device)
        validation_time = time.time() - validation_starting_time
        perplexity = math.exp(min(val_loss, 20))
        print(f"Epoch {epoch}: train_loss={train_avg}, val_loss={val_loss},  val_perplexity={perplexity}, training_time={training_time}s,  validation_time={validation_time}s")

        #Save checkpoint
        path = save_checkpoint(model, optimizer, epoch, global_step, examples_seen, checkpoint_directory)
        print(f"saved {os.path.basename(path)}")

    print("Training done")

if __name__ == "__main__":
    main()