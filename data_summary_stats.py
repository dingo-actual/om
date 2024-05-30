import json

from src.data.util import get_dataset_statistics

starcoder_dir = "/home/ubuntu/om-data/starcoderdata"
starcoder_match = ".parquet"

slimpajama_dir = "/home/ubuntu/om-data/SlimPajama-627B/train"
slimpajama_match = ".jsonl.zst"

starcoder_stats = get_dataset_statistics(starcoder_dir, starcoder_match, column="content")
slimpajama_stats = get_dataset_statistics(slimpajama_dir, slimpajama_match, field="text")

dataset_stats = {
    "starcoder": starcoder_stats,
    "slimpajama": slimpajama_stats
}

with open("dataset_stats.json", "w") as f:
    json.dump(dataset_stats, f)
