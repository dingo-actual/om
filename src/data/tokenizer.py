import tiktoken

enc_base = tiktoken.get_encoding("cl100k_base")
enc = tiktoken.Encoding(
    name="cl100k_base",
    pat_str=enc_base._pat_str,
    mergeable_ranks=enc_base._mergeable_ranks,
    special_tokens={
        **enc_base._special_tokens,
        "<|im_start|>": 100264,
        "<|im_end|>": 100265,
    }
)