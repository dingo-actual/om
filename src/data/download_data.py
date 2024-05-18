import json
from os.path import join
from pathlib import Path
from typing import Any

import tiktoken

from datasets import load_dataset


def save_dataset(dataset: Any, dataset_key: str, enc: tiktoken.Encoding, save_path: str) -> None:
    """Save a dataset to disk, along with token counts per file.

    Args:
        dataset (Any): Iterable dataset.
        enc (tiktoken.Encoding): Tokenizer.
        save_path (str): Path to save the dataset.
    """
    
    # Initialize counters
    file_num = 0
    sample_num = 0
    total_tokens = 0
    file_tokens = 0
    
    # Iterate over full dataset
    for sample in dataset["train"]:
        # Write tokenized sample to current file
        with open(join(save_path, f"{file_num:10d}.jsonl"), "a", encoding="utf-8") as fp:
            encoded = enc.encode(sample[dataset_key], allowed_special="all")
            json.dump(encoded, fp)
            fp.write("\n")
            
            # Increment file_tokens and total_tokens
            file_tokens += len(encoded)
            total_tokens += len(encoded)
            
        # Increment sample_num
        sample_num += 1
        
        # When 1000 samples have been written to current file, increment file
        if sample_num >= 1000:
            # Write token count for current file to token counts file
            with open(join(save_path, "token_counts.txt"), "a", encoding="utf-8") as fp:
                fp.write(f"{file_tokens}\n")
                
            # Print progress
            print(f"Saved 1000 samples ({file_tokens} tokens) to file: {file_num:10d}.jsonl")
            
            # Reset counters
            file_tokens = 0
            sample_num = 0
            
            # Increment file counter
            file_num += 1
    
    # If there are any remaining samples, update token counts and print progress message
    if sample_num > 0:
        with open(join(save_path, "token_counts.txt"), "a", encoding="utf-8") as fp:
            fp.write(f"{file_tokens}\n")

        print(f"Saved {sample_num} samples ({file_tokens} tokens) to file: {file_num:10d}.jsonl")
    
    # Write total token count to disk
    with open(join(save_path, "total_tokens.txt"), "w", encoding="utf-8") as fp:
        fp.write(str(total_tokens))
    
    # Print final progress message
    print(f"\nProcessing complete. Total tokens saved: {total_tokens:,d}\n\n")


if __name__ == "__main__":
    # Instantiate tokenizer
    enc_base = tiktoken.get_encoding("cl100k_base")
    enc = tiktoken.Encoding(
        name="cl100k_base",
        pat_str=enc_base._pat_str,
        mergeable_ranks=enc_base._mergeable_ranks
    )

    # Load datasets
    dataset_slimpajama = load_dataset("cerebras/SlimPajama-627B", streaming=True)
    dataset_starcoder = load_dataset("bigcode/starcoderdata", streaming=True)
    
    # Define dataset save paths
    save_path_slimpajama = join(Path(__file__).parent, "slimpajama")
    save_path_starcoder = join(Path(__file__).parent, "starcoder")
    
    # Save datasets to disk
    save_dataset(
        dataset=dataset_slimpajama, 
        dataset_key="text", 
        enc=enc, 
        save_path=save_path_slimpajama
    )
    save_dataset(
        dataset=dataset_starcoder, 
        dataset_key="content", 
        enc=enc, 
        save_path=save_path_starcoder
    )
