from torch.utils.data import DataLoader

from ..data.util import get_datasets_stages
from ..data.tokenizer import enc


def test_data_load():
    datasets = get_datasets_stages(
        dirs=[
            "/home/ubuntu/om-data/starcoderdata",
            "/home/ubuntu/om-data/SlimPajama-627B/train"
        ],
        matches=[
            ".parquet",
            ".jsonl.zst"
        ],
        datasets_num_files=[
            [43, 2958],
            [86, 11833],
            [215, 17749],
            [519, 26626]
        ],
        segment_lens=[
            128,
            1024,
            8192,
            131072
        ],
        batch_sizes=[
            512,
            128,
            32,
            2
        ],
        batch_proportions=[
            [387, 125],
            [110, 18],
            [25, 7],
            [1, 1]
        ],
        dataset_types=[
            "parquet",
            "jsonl"
        ],
        start_str="<|im_start|>",
        end_str="<|im_end|>",
        pad_str="<|pad|>",
        tokenizer=enc,
        field="text",
        column="content"
    )
    
    batch_sizes = [
        512,
        128,
        32,
        2
    ]
    
    dataloaders = [
        DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1
        )
        for dataset, batch_size in zip(datasets, batch_sizes)
    ]
    
    total_tokens = 0
    
    for ix, dataloader in enumerate(dataloaders):
        total_tokens_dataloader = 0
        for batch in dataloader:
            total_tokens_dataloader += batch.size(0) * batch.size(1)
        print(f"Dataloader {ix} completed successfully. Total tokens: {total_tokens_dataloader:,d}")
        total_tokens += total_tokens_dataloader
        
    print(f"All dataloaders completed successfully. Total tokens: {total_tokens:,d}")
