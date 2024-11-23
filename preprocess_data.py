import argparse
from os import listdir, mkdir
from os.path import join
from typing import List

import orjson

from src.om.data.tokenizer import enc


class ChunkFileWriter:
    def __init__(self, out_dir: str, chunk_size: int):
        """Initialize the ChunkFileWriter."""
        self.out_dir = out_dir
        self.file_num = 0
        
        self.chunk_size = chunk_size
        self.chunk_data = []
        
        self.tokens_written = dict()
        
    def add(self, data: List[int]):
        """Add data to the current chunk."""
        self.chunk_data.append(data)

        if len(self.chunk_data) >= self.chunk_size:
            self.flush()
            
    def flush(self):
        """Flush any remaining data to disk."""
        if len(self.chunk_data) > 0:
            fname = f"{self.file_num:010d}.json"
            fpath_out = join(self.out_dir, fname)
            with open(fpath_out, "wb") as fp:
                fp.write(orjson.dumps(self.chunk_data))

            self.tokens_written[fname] = sum([len(seq) for seq in self.chunk_data])

            self.file_num += 1
            self.chunk_data = []


def preprocess_data(
    data_dir: str, 
    out_dir: str, 
    cutoffs: List[int], 
    chunk_size: int,
    prepend_str: str = "",
    append_str: str = "",
    num_pad: int = 0,
):
    """Preprocess the data."""
    writers = []
    cutoff_prev = 0
    
    cutoffs = cutoffs + [float("inf")]
    
    for ix, cutoff in enumerate(cutoffs):
        if ix < len(cutoffs) - 1:
            cutoff_str = f"{cutoff_prev}-{cutoff}"
        else:
            cutoff_str = f"{cutoff_prev}+"
        range_out_dir = join(out_dir, cutoff_str)
        mkdir(range_out_dir)
        writers.append(ChunkFileWriter(range_out_dir, chunk_size))
        
        cutoff_prev = cutoff

    fnames = listdir(data_dir)

    for fname in fnames:
        fpath = join(data_dir, fname)
        with open(fpath, "rb") as fp:
            data = orjson.loads(fp.read())
            
        data = [prepend_str + line + append_str for line in data]
        data_enc = enc.encode_batch(data, allowed_special={"<|im_start|>", "<|im_end|>", "<|pad|>"})
        
        for line_enc in data_enc:
            for ix, cutoff in enumerate(cutoffs):
                if len(line_enc) <= cutoff - num_pad:
                    writers[ix].add(line_enc)
                    break
    
    tokens_summary = dict()
    
    prev_cutoff = 0
    for ix, (writer, cutoff) in enumerate(zip(writers, cutoffs)):
        writer.flush()

        if ix < len(cutoffs) - 1:
            cutoff_str = f"{prev_cutoff}-{cutoff}"
        else:
            cutoff_str = f"{prev_cutoff}+"
        prev_cutoff = cutoff
            
        tokens_summary[cutoff_str] = {
            "tokens_per_file": writer.tokens_written,
            "tokens_total": sum(writer.tokens_written.values()),
        }
        
    with open(join(out_dir, "tokens_summary.json"), "wb") as fp:
        fp.write(orjson.dumps(tokens_summary))
        
        
parser = argparse.ArgumentParser(
    prog="Preprocess data", 
    description="Tokenizes data, sorting by sequence length ranges."
)
parser.add_argument(
    "configs_fpath",
    action="store",
    type=str,
    nargs=1,
    help="Path to the configs file.",
)


if __name__=="__main__":
    args = parser.parse_args()
    with open(args.configs_fpath[0], "rb") as fp:
        configs = orjson.loads(fp.read())

    for config in configs:
        preprocess_data(
            data_dir=config["data_dir"],
            out_dir=config["out_dir"],
            cutoffs=config["cutoffs"],
            chunk_size=config["chunk_size"],
            prepend_str=config["prepend_str"],
            append_str=config["append_str"],
            num_pad=config["num_pad"],
        )
