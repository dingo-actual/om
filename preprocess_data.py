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
        
    def add(self, data: List[int]):
        """Add data to the current chunk."""
        self.chunk_data.append(data)

        if len(self.chunk_data) >= self.chunk_size:
            self.flush()
            
    def flush(self):
        """Flush any remaining data to disk."""
        if len(self.chunk_data) > 0:
            fpath_out = join(self.out_dir, f"{self.file_num:010d}.json")
            with open(fpath_out, "wb") as fp:
                fp.write(orjson.dumps())

            self.file_num += 1
            self.chunk_data = []


def preprocess_data(
    data_dir: str, 
    out_dir: str, 
    cutoffs: List[int], 
    chunk_size: int,
    prepend_str: str = "",
    append_str: str = "",
    pad_str: str = "",
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
        if num_pad > 0 and pad_str != "":
            data = [[num_pad] * num_pad + line for line in data]
        data_enc = enc.encode_batch(data, allowed_special={"<|im_start|>", "<|im_end|>", "<|pad|>"})
        
        for line_enc in data_enc:
            for ix, cutoff in enumerate(cutoffs):
                if len(line_enc) <= cutoff:
                    writers[ix].add(line_enc)
                    break
                
    for writer in writers:
        writer.flush()
