"""
What this does:
- creates a 95/5 train/val split
- makes a byte pair tokenizer based on the training data
- tokenizes each example to a fixed length sequence like this: [BOS, abstract..., SEP, title..., EOS, PAD, PAD, ...]
- saves the data
- saves vocab.json, merges.txt, train_tokens.bin, train_mask.bin, val_tokens.bin, val_mask.bin
"""

"""
before running this, run:

!pip install tokenizers

from google.colab import drive
drive.mount("/content/drive", force_remount=True)
"""

"""
Full disclosure, I had an LLM help me make this. That said, I did my best to make sure its understandable / that I understand it, and I hope that shows through in my discussion of what's happening throughout. I also made sure it's doing things I agree with / think make sense for this project. This is not a "prompt the LLM, ctrl+C, ctrl+V, call it a day" project. At the beginning of class, you said we should understand our code inside and out, and I think I do despite the LLM assistance.
"""

import json
import random
import time
from tokenizers import ByteLevelBPETokenizer
import numpy as np

#config
jsonl_path = "/content/drive/MyDrive/MSAI/spring_2026_quarter/GenAI/text_generation/dataset1/patents.jsonl"
prepared_data_directory = "/content/drive/MyDrive/MSAI/spring_2026_quarter/GenAI/text_generation/dataset1/prepared_data"

intended_vocab_size = 8000 #must match up in train and generate files. Also note that this can differ from the final vocab size because of the min byte pair frequency
block_size = 512 #must match up in train and generate files
min_byte_pair_frequency = 2

#this all must match up in train and generate files
pad_token = "<pad>"
bos_token = "<bos>"
sep_token = "<sep>"
eos_token = "<eos>"
special_tokens = [pad_token, bos_token, sep_token, eos_token]

val_set_split_ratio = 1 / 20  #for 5% train/val split
split_seed = 33


def main():
    #read JSONL
    print(f"Reading {jsonl_path}...")
    title_abstract_pairs = []   #(abstract, title) list
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            abstract = rec.get("abstract") or ""
            title = rec.get("title") or ""
            pn = rec.get("patent_number") or ""
            if abstract and title and pn:
                title_abstract_pairs.append((abstract, title))
    print(f"loaded {len(title_abstract_pairs)} title abstract pairs")

    #shuffle then 5% to validation set
    random.seed(split_seed)
    random.shuffle(title_abstract_pairs)
    n_val = int(len(title_abstract_pairs) * val_set_split_ratio)
    val_split = title_abstract_pairs[:n_val]
    train_split = title_abstract_pairs[n_val:]
    print(f"  train: {len(train_split):,}, val: {len(val_split):,}")

    print("Training byte-level BPE tokenizer...")
    t0 = time.time()

    def text_iter():
        for abstract, title in train_split:
            yield abstract
            yield title

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        text_iter(),
        vocab_size=intended_vocab_size,
        min_frequency=min_byte_pair_frequency,
        special_tokens=special_tokens,
    )
    tokenizer.save_model(prepared_data_directory)
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"  done in {time.time() - t0}s; vocab size = {actual_vocab_size}")

    pad_id = tokenizer.token_to_id(pad_token)
    bos_id = tokenizer.token_to_id(bos_token)
    sep_id = tokenizer.token_to_id(sep_token)
    eos_id = tokenizer.token_to_id(eos_token)
    print(f"special token IDs: pad={pad_id} bos={bos_id} sep={sep_id} eos={eos_id}")

    #tokenize and write to bin files
    def encode_example(abstract, title):
        abstract_ids = tokenizer.encode(abstract).ids
        title_ids = tokenizer.encode(title).ids
        if len(title_ids) + 4 > block_size: #This is for exceptionally long titles which should never happen but might as well (I dont know the PTO rule for this off the top of my head or if it changed).
            print(
                f"Dropping example: title too long. "
                f"title_tokens={len(title_ids)}, block_size={block_size}, "
                f"minimum_needed={len(title_ids) + 4}, "
                f"title_preview={title[:100]}"
            )
            return None
        max_abstract = block_size - 3 - len(title_ids)
        if len(abstract_ids) > max_abstract:
            abstract_ids = abstract_ids[-max_abstract:]  #if abstract is too long, left truncate it.
        seq = [bos_id] + abstract_ids + [sep_id] + list(title_ids) + [eos_id]
        sep_pos = 1 + len(abstract_ids)

        #masking out the title and eos tokens
        mask = [0] * block_size
        for i in range(sep_pos, sep_pos + len(title_ids) + 1):
            mask[i] = 1
        seq.extend([pad_id] * (block_size - len(seq)))
        return seq, mask

    def tokenize_split(pairs, name):
        n = len(pairs)
        tokens_array = np.zeros((n, block_size), dtype=np.uint16)
        mask_array = np.zeros((n, block_size), dtype=np.uint8)
        kept = 0
        dropped = 0
        t0 = time.time()
        for i, (abstract, title) in enumerate(pairs):
            result = encode_example(abstract, title)
            if result is None: #dropping the long title ones from above
                dropped += 1
                continue
            seq, mask = result
            tokens_array[kept] = seq
            mask_array[kept] = mask
            kept += 1
            if (i + 1) % 20000 == 0:
                rate = (i + 1) / max(time.time() - t0, 1e-6)
                print(f"  {name}: {i + 1}/{n} processed, kept {kept}, rate {rate}/s")
        tokens_array = tokens_array[:kept]
        mask_array = mask_array[:kept]
        tokens_array.tofile(f"{prepared_data_directory}/{name}_tokens.bin")
        mask_array.tofile(f"{prepared_data_directory}/{name}_mask.bin")
        print(f"done writing {name} bin file")
        return kept

    print("Tokenizing train split...")
    n_train_final = tokenize_split(train_split, "train")
    print("Tokenizing val split...")
    n_val_final = tokenize_split(val_split, "val")

    print(f"Done.")
    print(f"train size: {n_train_final}, val size: {n_val_final}")
    print(f"vocab size: {actual_vocab_size}, block size: {block_size}")

if __name__ == "__main__":
    main()