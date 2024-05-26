from io import TextIOWrapper
from typing import Generator, List

import orjson
import polars as pl
from tiktoken import Encoding
import torch
from torch.utils.data import Dataset
import zstandard as zstd


class FilesDataset(Dataset):
    def __init__(self, fpaths: List[str], segment_len: int):
        """Initialize dataset.

        Args:
            fpaths (List[str]): List of paths to the files.
            segment_len (int): Length of the segments for each sub-sample.
        """
        super(FilesDataset, self).__init__()
        
        self.fpaths = fpaths
        self.segment_len = segment_len

        self.buffer = []
        
        self.current_file_ix = 0
        self.current_obs_ix = 0
        
        self.open_current_file()
        
        self.current_file_ix += 1

    def open_current_file(self):
        raise NotImplementedError
            
    def __iter__(self):
        """Iterate over the dataset."""
        return self
    
    def __next__(self) -> torch.Tensor:
        """Return the next sample in the dataset.
        
        Returns:
            str: Current sample.
        """
        # Fill the buffer if necessary
        while len(self.buffer) < self.segment_len:
            # Read next line from current file
            if self.current_obs_ix == len(self.current_file):
                # If no more lines in current file, open next file
                self.current_file_ix += 1
                self.current_obs_ix = 0

                # If no more files, raise StopIteration
                if self.current_file_ix == len(self.fpaths):
                    raise StopIteration
                else:
                    self.open_current_file()
            else:
                # Add next line to buffer
                self.buffer = self.buffer + self.current_file[self.current_obs_ix]
                self.current_obs_ix += 1
                
        out = self.buffer[:self.segment_len]
        self.buffer = self.buffer[self.segment_len:]
                
        return torch.tensor(out)

class ParquetFilesDataset(FilesDataset):
    def __init__(
        self, 
        fpaths: List[str], 
        segment_len: int, 
        column: str, 
        start_str: str, 
        end_str: str, 
        tokenizer: Encoding
    ):
        """Initialize dataset.

        Args:
            fpaths (List[str]): List of paths to the parquet files.
            segment_len (int): Length of the segments for each sample.
            column (str): Name of column to use as input.
            start_str (str): Start string for the tokenizer.
            end_str (str): End string for the tokenizer.
            tokenizer (Encoding): Tokenizer to use.

        Raises:
            StopIteration: Reached end of dataset.
        """
        super(ParquetFilesDataset, self).__init__(fpaths, segment_len)
        self.column = column
        self.start_str = start_str
        self.end_str = end_str
        self.tokenizer = tokenizer
        
    def open_current_file(self):
        """Open the next parquet file."""
        self.current_file = (
            pl.read_parquet(
                self.fpaths[self.current_file_ix]
            )
            .get_column(self.column)
            .to_list()
        )
        self.current_file = [
            self.tokenizer.encode(f"{self.start_str}{line}{self.end_str}")
            for line in self.current_file
        ]
        self.current_file_ix += 1

class CompressedJSONLFilesDataset(FilesDataset):
    def __init__(
        self, 
        fpaths: List[str], 
        segment_len: int, 
        field: str, 
        start_str: str, 
        end_str: str, 
        tokenizer: Encoding
    ):
        """Initialize dataset.

        Args:
            fpaths (List[str]): List of paths to the compressed JSONL files.
            segment_len (int): Length of the segments for each sample.
            field (str): Field to extract from JSONL.
            start_str (str): Start string for the tokenizer.
            end_str (str): End string for the tokenizer.
            tokenizer (Encoding): Tokenizer to use.

        Raises:
            StopIteration: Reached end of dataset.
        """
        super(CompressedJSONLFilesDataset, self).__init__(fpaths, segment_len)
        self.field = field
        self.start_str = start_str
        self.end_str = end_str
        self.tokenizer = tokenizer
        
    def open_current_file(self):
        """Open the next compressed JSONL file."""
        self.current_file = list(self.read_zst_jsonl(self.fpaths[self.current_file_ix], self.field))
        self.current_file = [
            self.tokenizer.encode(f"{self.start_str}{line}{self.end_str}")
            for line in self.current_file
        ]
        self.current_file_ix += 1
        
    @staticmethod
    def read_zst_jsonl(fpath: str, field: str) -> Generator[str, None, None]:
        """Reads a compressed JSONL file.

        Args:
            fpath (str): Path to the compressed JSONL file.
            field (str): Field to extract from JSONL.

        Yields:
            Generator[str]: Output strings.
        """
        with open(fpath, "rb") as file:
            dctx = zstd.ZstdDecompressor()
            stream_reader = dctx.stream_reader(file)
            text_wrapper = TextIOWrapper(stream_reader, encoding="utf-8")
            for line in text_wrapper:
                yield orjson.loads(line)[field]

class ProportionalDataset(Dataset):
    def __init__(self, datasets: List[Dataset], proportions: List[int]) -> None:
        """Initialize dataset.

        Args:
            datasets (List[Dataset]): Datasets to sample from.
            proportions (List[int]): List of how many samples to take from each dataset.
        """
        super().__init__()
        
        self.datasets = datasets
        self.proportions = proportions
        self.current_dataset_ix = 0
        self.current_sample_ix = 0
        
        self.skip = [False if p > 0 else True for p in self.proportions]
        
    def __iter__(self):
        """Iterate over the dataset."""
        return self
    
    def __next__(self) -> torch.Tensor:
        """Return the next sample in the dataset.

        Returns:
            torch.Tensor: Current sample.
        """
        # If the current sample index is equal to the proportion for the current dataset,
        # increment the current dataset index and reset the current sample index
        if self.current_sample_ix >= self.proportions[self.current_dataset_ix] or self.skip[self.current_dataset_ix]:
            self.current_dataset_ix = (self.current_dataset_ix + 1) % len(self.datasets)
            self.current_sample_ix = 0

        # Try to sample from the current dataset
        try:
            sample = next(self.datasets[self.current_dataset_ix])
        # If the dataset is empty, set its proportion to 0 (so it'll get skipped)
        # If all datasets are empty, raise StopIteration
        except StopIteration:
            self.skip[self.current_dataset_ix] = True
            if all(self.skip):
                raise StopIteration
            else:
                return self.__next__()
        else:
            # If sampling from the current dataset was successful, increment the current sample index
            self.current_sample_ix += 1

        return sample
