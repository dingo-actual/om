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
            [8, 2959],
            [122, 11834],
            [216, 17749],
            [518, 26621]
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
            [882, 31],
            [202, 9],
            [5, 3]
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
    
    for ix, dataloader in enumerate(dataloaders):
        total_tokens_dataloader = 0
        for batch in dataloader:
            total_tokens_dataloader += batch.size(0) * batch.size(1)
        print(f"Dataloader {ix} completed successfully. Total tokens: {total_tokens_dataloader:,d}")
        total_tokens += total_tokens_dataloader
        
    print(f"All dataloaders completed successfully. Total tokens: {total_tokens:,d}")
