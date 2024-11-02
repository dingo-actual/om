import tiktoken

enc_base = tiktoken.get_encoding("gpt2")
enc = tiktoken.Encoding(
    name="gpt2",
    pat_str=enc_base._pat_str,
    mergeable_ranks=enc_base._mergeable_ranks,
    special_tokens={
        "<|im_start|>": 50281,
        "<|im_end|>": 50256,
        "<|pad|>": 50282,
    }
)
