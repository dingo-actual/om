import json
from os.path import join
from pathlib import Path

import tiktoken

from datasets import load_dataset

enc_base = tiktoken.get_encoding("cl100k_base")
enc = tiktoken.Encoding(
    name="cl100k_base",
    pat_str=enc_base._pat_str,
    mergeable_ranks=enc_base._mergeable_ranks,
    special_tokens={
        **enc_base._special_tokens,
        "<|im_start|>": 100264,
        "<|im_end|>": 100265,
        "<|pad|>": 100263
    }
)

dataset_starcoder = load_dataset("bigcode/starcoderdata", streaming=True)
dataset_slimpajama = load_dataset("cerebras/SlimPajama-627B", streaming=True)


def save_starcoder_data(save_path: str) -> None:
    file_num = 0
    sample_num = 0
    total_tokens = 0
    for sample in dataset_starcoder["train"]:
        with open(join(save_path, f"data_{file_num:10d}.jsonl"), "a", encoding="utf-8") as f:
            encoded = enc.encode(sample["text"], allowed_special="all")
            json.dump(encoded, f)
            f.write("\n")
            total_tokens += len(encoded)
        sample_num += 1
        
        if sample_num >= 1000:
            sample_num = 0
            file_num += 1
            print(f"Saved 1000 samples to file: slimpajama_data_{file_num:10d}.jsonl")
            
    if sample_num > 0:
        with open(join(save_path, f"data_{file_num:10d}.jsonl"), "a", encoding="utf-8") as f:
            encoded = enc.encode(sample["text"], allowed_special="all")
            json.dump(encoded, f)
            f.write("\n")
            total_tokens += len(encoded)
        print(f"Saved {sample_num} samples to file: starcoder_data_{file_num:10d}.json")
            
    with open(join(save_path, "total_tokens.txt"), "w", encoding="utf-8") as f:
        f.write(str(total_tokens))
        
def save_slimpajama_data(save_path: str) -> None:
    file_num = 0
    sample_num = 0
    total_tokens = 0
    for sample in dataset_starcoder["train"]:
        with open(join(save_path, f"data_{file_num:10d}.json"), "a", encoding="utf-8") as f:
            encoded = enc.encode(sample["text"], allowed_special="all")
            json.dump(encoded, f)
            f.write("\n")
            total_tokens += len(encoded)
        sample_num += 1
        
        if sample_num >= 1000:
            sample_num = 0
            file_num += 1
            print(f"Saved 1000 samples to file: starcoder_data_{file_num:10d}.json")
    
    if sample_num > 0:
        with open(join(save_path, f"data_{file_num:10d}.jsonl"), "a", encoding="utf-8") as f:
            encoded = enc.encode(sample["text"], allowed_special="all")
            json.dump(encoded, f)
            f.write("\n")
            total_tokens += len(encoded)
        print(f"Saved {sample_num} samples to file: starcoder_data_{file_num:10d}.json")
    
    with open(join(save_path, "total_tokens.txt"), "w", encoding="utf-8") as f:
        f.write(str(total_tokens))


if __name__ == "__main__":
    save_path_slimpajama = join(Path(__file__).parent, "slimpajama")
    save_path_starcoder = join(Path(__file__).parent, "starcoder")
    
    save_slimpajama_data(save_path=save_path_slimpajama)
    save_starcoder_data(save_path=save_path_starcoder)
