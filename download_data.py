import argparse

from datasets import load_dataset

from src.data import enc, save_dataset


parser = argparse.ArgumentParser(
    prog="download_data.py",
    description="Download and tokenize a dataset, saving the results to disk."
)
parser.add_argument(
    "dataset", 
    nargs=1,
    type=str, 
    choices=["slimpajama", "starcoder"],
    help="Dataset to download and tokenize."
)
parser.add_argument(
    "split",
    nargs=1,
    type=str,
    choices=["train", "validation", "test"],
    help="Split of the dataset to download and tokenize."
)
parser.add_argument(
    "key",
    nargs=1,
    type=str,
    help="Key for each sample in the dataset."
)
parser.add_argument(
    "save_path",
    nargs=1,
    type=str,
    help="Path to save the dataset to."
)

dataset_map = {
    "slimpajama": "cerebras/SlimPajama-627B",
    "starcoder": "bigcode/starcoderdata"
}


if __name__ == "__main__":
    args = vars(parser.parse_args())
    
    dataset = args["dataset"][0]
    dataset_name = dataset_map[dataset]
    split = args["split"][0]
    key = args["key"][0]
    save_path = args["save_path"][0]
    
    save_dataset(
        dataset=load_dataset(dataset_name, streaming=True),
        split=split,
        dataset_key=key,
        enc=enc,
        save_path=save_path
    )
