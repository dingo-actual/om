from time import time

import numpy as np
from torch.utils.data import DataLoader

from ..data.load import get_datasets_stages
from ..data.tokenizer import enc


def test_data_load():
    batch_sizes = [
        1024,
        128,
        32,
        8
    ]
    
    datasets = get_datasets_stages(
        dirs=[
            "/home/ubuntu/om-data/data_preprocessed/starcoderdata",
            "/home/ubuntu/om-data/data_preprocessed/SlimPajama-627B/train"
        ],
        matches=[
            ".jsonl",
            ".jsonl"
        ],
        datasets_num_files=[
            [8, 2958],
            [43, 11833],
            [86, 17749],
            [172, 26624]
        ],
        segment_lens=[
            128,
            1024,
            8192,
            131072
        ],
        batch_sizes=batch_sizes,
        batch_proportions=[
            [939, 85],
            [114, 14],
            [27, 5],
            [7, 1]
        ]
    )
    
    dataloaders = [
        DataLoader(
            dataset=dataset,
            batch_size=batch_size // 8,
            num_workers=1
        )
        for dataset, batch_size in zip(datasets, batch_sizes)
    ]
    
    total_tokens = 0
    
    times = []
    for ix, dataloader in enumerate(dataloaders):
        total_tokens_dataloader = 0
        for batch in dataloader:
            times.append(time())
            total_tokens_dataloader += batch.size(0) * batch.size(1)
        print(f"Dataloader {ix+1} completed successfully. Total tokens: {total_tokens_dataloader:,d}")
        break
        total_tokens += total_tokens_dataloader
        
    print(f"All dataloaders completed successfully. Total tokens: {total_tokens:,d}")
    time_diffs = np.array([t2 - t1 for t1, t2 in zip(times[:-1], times[1:])])
    
    print(f"Average time per batch: {time_diffs.mean():.6f}s")
    print(f"Median time per batch: {np.median(time_diffs):.6f}s")
    print(f"99th percentile time per batch: {np.percentile(time_diffs, q=99):.6f}s")
    print(f"Max time per batch: {time_diffs.max():.6f}s")
    