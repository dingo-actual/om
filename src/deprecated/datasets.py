from io import TextIOWrapper
from typing import Generator, List

import orjson
import polars as pl
from tiktoken import Encoding
from torch.utils.data import IterableDataset
import zstandard as zstd

from ..data.datasets import FilesDataset

class ParquetFilesDataset(FilesDataset):
    def __init__(
        self, 
        fpaths: List[str], 
        segment_len: int, 
        column: str, 
        start_str: str, 
        end_str: str, 
        pad_str: str,
        tokenizer: Encoding,
        **kwargs
    ):
        """Initialize dataset.

        Args:
            fpaths (List[str]): List of paths to the parquet files.
            segment_len (int): Length of the segments for each sample.
            column (str): Name of column to use as input.
            start_str (str): Start string for the tokenizer.
            end_str (str): End string for the tokenizer.
            pad_str (str): Padding string for the tokenizer.
            tokenizer (Encoding): Tokenizer to use.

        Raises:
            StopIteration: Reached end of dataset.
        """
        super(ParquetFilesDataset, self).__init__(fpaths, segment_len, start_str, end_str, pad_str, tokenizer)
        self.column = column
        
    def open_current_file(self):
        """Open the next parquet file."""
        self.current_file = (
            pl.read_parquet(
                self.fpaths[self.current_file_ix]
            )
            .get_column(self.column)
            .to_list()
        )
        self.current_file_ix += 1

class CompressedJSONLFilesDataset(FilesDataset):
    def __init__(
        self, 
        fpaths: List[str], 
        segment_len: int, 
        field: str, 
        start_str: str, 
        end_str: str, 
        pad_str: str,
        tokenizer: Encoding,
        **kwargs
    ):
        """Initialize dataset.

        Args:
            fpaths (List[str]): List of paths to the compressed JSONL files.
            segment_len (int): Length of the segments for each sample.
            field (str): Field to extract from JSONL.
            start_str (str): Start string for the tokenizer.
            end_str (str): End string for the tokenizer.
            pad_str (str): Padding string for the tokenizer.
            tokenizer (Encoding): Tokenizer to use.

        Raises:
            StopIteration: Reached end of dataset.
        """
        super(CompressedJSONLFilesDataset, self).__init__(fpaths, segment_len, start_str, end_str, pad_str, tokenizer)
        self.field = field
        
    def open_current_file(self):
        """Open the next compressed JSONL file."""
        self.current_file = list(self.read_zst_jsonl(self.fpaths[self.current_file_ix], self.field))
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