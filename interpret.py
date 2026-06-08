"""
What this does:
- generates a patent title from one abstract (using greedy decoding by default)
- for each generated title token, computes |gradient * input| saliency
  showing which input tokens from (BOS, abstract tokens, SEP) most influenced
  the model's choice of that title token
- saves a heatmap PNG and prints the top-K most-influential input tokens
  for each generated title token
"""

"""
before running this, run:

!pip install transformers tokenizers

from google.colab import drive
drive.mount("/content/drive", force_remount=True)
"""

"""
See LLM note at the top of tokenize_data.py
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Config, GPT2LMHeadModel
from tokenizers import ByteLevelBPETokenizer

#paths
prepared_data_directory = "/content/drive/MyDrive/MSAI/spring_2026_quarter/GenAI/text_generation/dataset1/prepared_data"
checkpoint_to_load = "/content/drive/MyDrive/MSAI/spring_2026_quarter/GenAI/text_generation/dataset1/checkpoints/ckpt_epoch_008.pt"
saliency_output_directory = "/content/drive/MyDrive/MSAI/spring_2026_quarter/GenAI/text_generation/dataset1/saliency"

#must match prepare_data.py and train.py
block_size = 512
max_new_tokens = 64

#special tokens (must match prepare_data.py)
pad_token = "<pad>"
bos_token = "<bos>"
sep_token = "<sep>"
eos_token = "<eos>"

#how many top-influencing input tokens to print per title token
top_k_per_title_token = 5

#abstract to analyze
abstract = (
    "Provided is a method capable of simply and exponentially amplifying circular DNA, and particularly, long-chain circular DNA, in a cell-free system. Specifically, provided herein is a method for amplifying circular DNA which comprises mixing circular DNA having a replication origin sequence (origin of chromosome (oriC)) with a reaction solution comprising: a first enzyme group that catalyzes replication of circular DNA; a second enzyme group that catalyzes an Okazaki fragment maturation and synthesizes two sister circular DNAs constituting a catenane; a third enzyme group that catalyzes a separation of two sister circular DNAs; and also, a buffer, NTP, dNTP, a magnesium ion source, and an alkali metal ion source, to form a reaction mixture, which is then reacted."
)


def load_model_from_checkpoint(checkpoint_path):
    print(f"Loading {os.path.basename(checkpoint_path)}")
    state = torch.load(checkpoint_path, map_location="cuda")
    config = GPT2Config(**state["config"])
    model = GPT2LMHeadModel(config).to("cuda")
    model.load_state_dict(state["model"])
    model.eval()
    print(f"Model loaded. Epoch={state['epoch']}, step={state['step']}")
    return model

@torch.no_grad()
def generate_title(model, tokenizer, abstract):
    bos_id = tokenizer.token_to_id(bos_token)
    sep_id = tokenizer.token_to_id(sep_token)
    eos_id = tokenizer.token_to_id(eos_token)
    pad_id = tokenizer.token_to_id(pad_token)
    abstract_ids = tokenizer.encode(abstract).ids

    #truncating the abstract if its too long for the context length
    max_abstract_length = block_size - 2 - max_new_tokens
    if len(abstract_ids) > max_abstract_length:
        abstract_ids = abstract_ids[-max_abstract_length:]

    prefix = [bos_id] + abstract_ids + [sep_id] #this is what the model starts from when generating a title
    input_ids = torch.tensor([prefix], dtype=torch.long, device="cuda")

    #generating the output
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=eos_id, pad_token_id=pad_id) #this is where greedy decoding is set

    #converting the output to something human readable
    generated_ids = output[0, len(prefix):].tolist()
    if generated_ids and generated_ids[-1] == eos_id:
        generated_ids = generated_ids[:-1]
    return prefix, generated_ids #note the return is different than generate

#the point here is to, for each title token, run the model and take the gradient of the logits with respect to the input embeddings and find the importance of those input embeddings to that output token
def compute_saliency(model, prefix, title_ids):
    n_prefix = len(prefix)
    n_title = len(title_ids)
    importance_array = np.zeros((n_title, n_prefix))

    for t in range(n_title):
        input_seq = prefix + title_ids[:t]
        target_token = title_ids[t]
        input_ids = torch.tensor([input_seq], dtype=torch.long, device="cuda")

        model.zero_grad()
        embeddings = model.transformer.wte(input_ids) #token embeddings. This makes a tensor of shape (1, tokens, embedding dimension)
        embeddings.retain_grad()

        outputs = model(inputs_embeds=embeddings) #running the post embedding part of the model
        score = outputs.logits[0, -1, target_token] #find the logit for the token of interest
        score.backward() #this updates embeddings.grad with the derivatives of score wrt embedding dimensions for each token
        importance = (embeddings.grad * embeddings).sum(dim=-1)[0] #this multiplies those derivatives by the embeddings to get the degree to which they contribute to the score for the token of interest
        importance_array[t] = importance.abs()[:n_prefix].detach().cpu().numpy() #drop irrelevant parts and make it a np array
    return importance_array

#ByteLevelBPE uses Ġ for leading spaces and Ċ for newlines for some reason so this makes it more readable
def label_for_token(tokenizer, token_id):
    raw = tokenizer.id_to_token(token_id)
    if raw is None:
        return f"<{token_id}>"
    return raw.replace("Ġ", " ").replace("Ċ", "↵")

def plot_heatmap(importance_array, prefix, title_ids, tokenizer, output_path):
    prefix_labels = [label_for_token(tokenizer, t) for t in prefix]
    title_labels = [label_for_token(tokenizer, t) for t in title_ids]

    fig_width = max(12, len(prefix_labels) * 0.18)
    fig_height = max(4, len(title_labels) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(importance_array, aspect="auto", cmap="viridis")

    ax.set_xticks(range(len(prefix_labels)))
    ax.set_xticklabels(prefix_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(title_labels)))
    ax.set_yticklabels(title_labels, fontsize=10)

    ax.set_xlabel("Input tokens (BOS, abstract..., SEP)")
    ax.set_ylabel("Generated title tokens")
    ax.set_title("|gradient × input| per input token, per generated title token")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"Saved heatmap to {output_path}")
    plt.show()


def print_top_k(importance_array, prefix, title_ids, tokenizer, k):
    print()
    print(f"Top {k} most-influential input tokens per generated title token:")
    for t, tid in enumerate(title_ids):
        tgt_label = label_for_token(tokenizer, tid)
        top_indices = np.argsort(importance_array[t])[::-1][:k]
        parts = [
            f"{label_for_token(tokenizer, prefix[i])!r}({importance_array[t][i]:.3f})"
            for i in top_indices
        ]
        print(f"  [{t}] {tgt_label!r}: " + ", ".join(parts))


def main():
    model = load_model_from_checkpoint(checkpoint_to_load)
    tokenizer = ByteLevelBPETokenizer(
        vocab=f"{prepared_data_directory}/vocab.json",
        merges=f"{prepared_data_directory}/merges.txt",
    )

    print()
    print("Abstract:")
    print(abstract)
    print()

    print("Generating title")
    prefix, title_ids = generate_title(model, tokenizer, abstract)
    title_text = tokenizer.decode(title_ids).strip()
    print(f"Generated title: {title_text}")
    print(f"({len(title_ids)} title tokens, {len(prefix)} prefix tokens)")
    print()

    importances = compute_saliency(model, prefix, title_ids)
    print("Done")

    os.makedirs(saliency_output_directory, exist_ok=True)
    output_path = f"{saliency_output_directory}/saliency_heatmap.png"
    plot_heatmap(importances, prefix, title_ids, tokenizer, output_path)
    print_top_k(importances, prefix, title_ids, tokenizer, top_k_per_title_token)

if __name__ == "__main__":
    main()