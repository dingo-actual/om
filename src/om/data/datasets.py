from typing import List

import orjson
from tiktoken import Encoding
import torch
from torch.utils.data import IterableDataset


class FilesDataset(IterableDataset):
    def __init__(
        self, 
        fpaths: List[str], 
        segment_len: int,
        enc: Encoding,
        prefix_str: str = "",
        suffix_str: str = "",
        pad_str: str = "",
        num_pad: int = 0
    ):
        """Initialize dataset.

        Args:
            fpaths (List[str]): List of paths to the files.
            segment_len (int): Length of the segments for each sub-sample.
            enc (Encoding): Tokenizer to use.
            prefix_str (str, optional): Prefix to add to each sample. Defaults to "".
            suffix_str (str, optional): Suffix to add to each sample. Defaults to "".
            pad_str (str, optional): String to use for pre-padding samples. Defaults to "".
            num_pad (int, optional): Number of pre-padding tokens. Defaults to 0.
        """
        super(FilesDataset, self).__init__()
        
        self.fpaths = fpaths
        self.segment_len = segment_len
        self.enc = enc
        self.prefix_str = prefix_str
        self.suffix_str = suffix_str
        self.pad_str = pad_str
        
        if prefix_str != "":
            self.prefix_seq = self.enc.encode(prefix_str)
        if suffix_str != "":
            self.suffix_seq = self.enc.encode(suffix_str)
        
        if num_pad <= 0 or pad_str == "":
            self.num_pad = 0
            self.pad_token = -1
            self.padding = None
        else:
            self.num_pad = num_pad
            self.pad_token = self.enc.encode_single_token(pad_str)
            self.padding = [self.pad_token] * self.num_pad

        self.buffer = []
        
        self.current_file_ix = 0
        self.current_obs_ix = 0
        
        _ = self.open_current_file()
        
        self.current_file_ix += 1

    def open_current_file(self) -> bool:
        """Open the next file."""
        if self.current_file_ix >= len(self.fpaths):
            return False
        
        with open(self.fpaths[self.current_file_ix], "rb") as fp:
            self.current_file = orjson.loads(fp.read())
        
        return True
    
    def reset(self):
        """Resets the dataset to its initial state."""
        self.buffer = []
        
        self.current_file_ix = 0
        self.current_obs_ix = 0
        
        _ = self.open_current_file()
        
        self.current_file_ix += 1
            
    def __iter__(self):
        """Iterate over the dataset."""
        return self
    
    def __next__(self) -> torch.Tensor:
        """Return the next sample in the dataset.
        
        Returns:
            torch.Tensor: Current sample.
        """
        # Fill the buffer if necessary
        while len(self.buffer) < self.segment_len - self.num_pad:
            # Read next line from current file
            if self.current_obs_ix == len(self.current_file):
                # If no more lines in current file, open next file
                self.current_file_ix += 1
                self.current_obs_ix = 0

                # Try to open current file; if no more files, raise StopIteration
                success = self.open_current_file()
                if not success:
                    self.reset()
                    raise StopIteration
            else:
                # Add next line to buffer
                if self.prefix_str != "":
                    self.buffer.extend(self.prefix_seq)
                self.buffer.extend(self.current_file[self.current_obs_ix])
                if self.suffix_str != "":
                    self.buffer.extend(self.suffix_seq)
                self.current_obs_ix += 1
        
        out = self.padding + self.buffer[:self.segment_len - self.num_pad]
        self.buffer = self.buffer[self.segment_len - self.num_pad:]
                
        return torch.tensor(out)

class ProportionalDataset(IterableDataset):
    def __init__(
        self, 
        datasets: List[IterableDataset], 
        proportions: List[int]
    ) -> None:
        """Initialize dataset.

        Args:
            datasets (List[IterableDataset]): Datasets to sample from.
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
                self.reset()
                raise StopIteration
            else:
                return self.__next__()
        else:
            # If sampling from the current dataset was successful, increment the current sample index
            self.current_sample_ix += 1

        return sample
    
    def reset(self):
        """Resets the dataset to its initial state"""
        self.current_dataset_ix = 0
        self.current_sample_ix = 0
        
        self.skip = [False if p > 0 else True for p in self.proportions]
        
        for dataset in self.datasets:
            dataset.reset()
