# Om LLM (README WIP)

<img src="om_llm.jpg" alt="Om LLM Logo" width="25%" height="25%">

## Overview

Om LLM is a project that implements an advanced large language model (LLM) architecture using Attentive Recurrent Cells (ARC).

### Features

- Arbitrary-length input sequences with bounded memory requirements
- Cache-able context
- Multi-pass memory transformer
- Ability to handle sequences of characters, or sequences of tokens
- Reduced dependence on tokenizer through use of initial convolutional layers

### Novel Contributions (at time of writing) and Publication

- Use of "state token sequences" in conjunction with a recurrence mechanism which utilizes attention calculations to update state
  - This allows for effective arbitrary-length input sequences
- Multi-pass transformer memory
  - This allows for greater expressivity and noise-robustness in the memory portion of the transformer
- Initial convolutional layers
  - This allows for the model to:
    - Learn effectively, regardless of the choice of tokenizer
    - Potentially eliminate the need for a tokenizer

To the best of my knowledge, these features are novel. If you find any references to these features in the literature, please let me know. My email is [ryan@beta-reduce.net](ryan@beta-reduce.net)

I do not plan on publishing a paper on this project. If you would like to use this project in your own work, please cite this repository, and its creator (Ryan P. Taylor).

## `ARC`: Attentive Recurrent Cell

The Attentive Recurrent Cell (`ARC`) is a lower-level component of Om LLM. It represents a recurrent cell that utilizes attention calculations to update state. Unlike the multi-head attention component of a transformer, the memory component of an `ARC` performs multiple project-then-attend operations. Beyond the first of these operations, the number of additional parameters added to the model is rather small, having nine matrices of size `(memory_dim_prev, memory_dim_next)` for each memory head: three for the "read" state sequence, three for the "write" state sequence, and three for the non-state sub-sequence. Additionally, `ARC` is compatible with RoPE and/or CoPE positional embeddings.

The forward pass of `ARC` requires three inputs:

- A `Tensor` sub-sequence of the input
- A current state token sequence `Tensor`, and
- An `int` offset, representing the location of the current sub-sequence in the input sequence.

It will produce two outputs `Tensor`s:

- The typical output sequence from a multi-head attention block, as well as
- A state token sequence, which is used to process the next input sub-sequence.

### `ARC` Usage

The `ARC` class can be instantiated with the following parameters:

- `dim_input` (`int`): The dimension of the input sequence.
- `dims_key` (`List[int]`): The dimensions of the key/query vectors for each layer in the attention block.
- `dims_value` (`List[int]`): The dimensions of the value vectors for each layer in the attention block.
- `num_heads` (`int`): The number of heads in the attention block.
- `segment_len` (`int`): The length of the segment to be processed at a time.
- `state_len` (`int`): The length of the state token sequence.
- `normalize` (`bool`): Whether to normalize the inputs to SDP attention.
- `num_layers` (`int`): The number of `ARCformer` layers in the parent `OmLLM` model (used for weight initialization).
- `cope` (`bool`): Whether to use CoPE positional embeddings.
- `position_embedders` (`List[Optional[RoPEEmmbeddings]]`): A list of optional positional embedding objects for each layer in the attention block.

Once instantiated, an `ARC` object can be called as follows:

```python
output, state_token_sequence = arc(input_sequence, state_token_sequence, offset)
```

The `output` `Tensor` will have the same shape as the input sequence, and can be passed to the MLP portion of an `ARCformer`. The `state_token_sequence` `Tensor` will have the same shape as `state_token_sequence`, and can be passed as the `state=` argument for the next sub-sequence processed by this `ARC` layer.

## `ARCformer`: Attentive Recurrent Cell (ARC) Transformer

The ARC Transformer (`ARCformer`) is a middle-level component of Om LLM. It represents a transformer-like architecture that incorporates multi-pass memory, as well as "state token sequences". The forward pass of `ARCformer` requires three inputs:

- A `Tensor` sub-sequence of the input
- A current state token sequence `Tensor` (the initial state for each `ARCformer` is a learned parameter), and
- An `int` offset, representing the location of the current sub-sequence in the input sequence.

It will produce two outputs `Tensor`s:

- The usual output sequence from a transformer block, as well as
- A state token sequence, which is used to process the next input sub-sequence.

### `ARCformer` Usage

The `ARCformer` class can be instantiated with the following parameters:

- `dim_input` (`int`): The dimension of the input sequence.
- `dim_hidden` (`int`): The dimension of the hidden layer for the MLP portion of the transformer.
- `dims_key` (`List[int]`): The dimensions of the key/query vectors for each layer in the attention block.
- `dims_value` (`List[int]`): The dimensions of the value vectors for each layer in the attention block.
- `num_heads` (`int`): The number of heads in the attention block.
- `activation` (`str`): The activation function to use for the MLP portion of the transformer. Must be one of:
  - "relu"
  - "gelu"
  - "swish"
  - "swiglu"
  - "geglu"
  - "ffnglu"
  - "ffngeglu"
  - "ffnswiglu"
  - "abs"
- `segment_len` (`int`): The length of the segment to be processed at a time.
- `state_len` (`int`): The length of the state token sequence.
- `normalize` (`bool`): Whether to normalize the inputs to SDP attention.
- `num_layers` (`int`): The number of `ARCformer` layers in the parent `OmLLM` model (used for weight initialization).
- `cope` (`bool`): Whether to use CoPE positional embeddings.
- `position_embedders` (`List[Optional[RoPEEmmbeddings]]`): A list of optional positional embedding objects for each layer in the attention block.
- `dropout` (`float`): The dropout rate for the MLP portion of the transformer. (Default: 0.0)
- `mlp_multiplier` (`int`): Multiplier for the final two layers of the MLP portion of the transformer. (Default: 1)

Once instantiated, an `ARCformer` object can be called as follows:

```python
output, state_token_sequence_next = arcformer(input_sequence, state_token_sequence, offset)
```

The `output` `Tensor` will have the same shape as the input sequence, and can be passed to the next `ARCformer` in the model. The `state_token_sequence_next` `Tensor` will have the same shape as `state_token_sequence`, and can be passed as the `state=` argument for the next sub-sequence processed by this `ARCformer`.

## `RoPEEmbeddings`

RoPEEmbeddings is a class that implements the RoPE (Rotary Position Embedding) positional embedding scheme, as described in the paper "RoFormer: Enhanced Transformer with Rotary Position Embedding" by Jianlin Su et al. ([arxiv](https://arxiv.org/abs/2104.09864)). It has minor modifications made to support `ARC`'s recurrent structure.

### `RoPEEmbeddings` Usage

The `RoPEEmbeddings` class can be instantiated with the following parameters:

- `dim` (`int`): Key/Query dimension of the corresponding attention layer.
- `seq_len` (`int`): Maximum sequence length.
- `dim_embedding_pct` (`float`): Percentage of the total embedding dimension to use for the positional embeddings. Must be within the interval (0, 1]. Defaults to 0.5.
- `base` (`int`, optional): Base used for calculating thetas. Defaults to 10000.

Once instantiated, a `RoPEEmbeddings` object can be called as follows:

```python
q_rope = rope(q, offset)
k_rope = rope(k, offset)

att = sdp_attention(q_rope, k_rope, v)
```

Although users are unlikely to directly interface with a `RoPEEmbeddings` object, it's necessary to instantiate them when using RoPE in an `OmLLM` object.

## `OmLLM`

The `OmLLM` class is the primary user-facing class in this package. It represents an LLM using the Om architecture, which utilizes `ARC` memory, as well as (optional) initial convolutional operations on the embeddings. Note that these embeddings can either come from tokens, or from direct characters.

The inputs to an `OmLLM` object are:

- A `Tensor` of token/character indices.
- A list of initial state token sequences (one for each `ARCformer` layer in the model). If an empty list is provided, the learned initial state token sequences will be used for each `ARCformer`. This argument is the means through which inference with a cached context is performed.
- An `int` offset for the input text (default: 0). This can be set to a nonzero value when performing inference with a cached context (where the value will be equal to the length of the cached context).
- A `bool` indicating whether or not to only produce predictions for the next token. If `False`, next-token predictions will be produced for the entire input.

The outputs of an `OmLLM` object are:

- A `Tensor` of logits for next tokens; this can either be a sequence of such logit vectors, or a single logit vector, depending on whether the user has specified to only predict the next token.
- A list of state token sequence `Tensor`s at the final input. This can be cached to perform context caching.
- An `int` offset for the input sequence. This can be used to perform context caching.

### `OmLLM` Usage

The `OmLLM` class can be instantiated with the following parameters:

- `num_layers` (`int`): The number of `ARCformer` layers in the model.
- `vocab_size` (`int`): The size of the input vocabulary. If not divisible by 8, this will be internally padded to the next multiple of 8 (to make softmax computation during next-token prediction more efficient).
- `dim_input` (`int`): The dimension of the input sequence.
- `dim_hidden` (`int`): The dimension of the hidden layer for the MLP portion of the transformers.
- `dims_key` (`List[int]`): The dimensions of the key/query vectors for each layer in the attention blocks.
- `dims_value` (`List[int]`): The dimensions of the value vectors for each layer in the attention blocks.
- `num_heads` (`int`): The number of heads in the attention blocks.
- `activation` (`str`): The activation function to use for the MLP portion of the transformers. Must be one of:
  - "relu"
  - "gelu"
  - "swish"
  - "swiglu"
  - "geglu"
  - "ffnglu"
  - "ffngeglu"
  - "ffnswiglu"
  - "abs"
- `segment_len` (`int`): The length of the segment to be processed at a time.
- `state_len` (`int`): The length of the state token sequence.
- `normalize` (`bool`): Whether to normalize the inputs to SDP attention.
- `cope` (`bool`): Whether to use CoPE positional embeddings.
- `position_embedders` (`List[Optional[RoPEEmmbeddings]]`): A list of optional positional embedding objects for each layer in the attention block.
- `dropout` (`float`): The dropout rate for the MLP portion of the transformer. (Default: 0.0)
- `init_convs` (`List[int]`): The kernel widths for initial convolutional layers. Leave empty to not use initial convolutional layers. (Default: [])
- `final_mlp_multiplier` (`int`): Multiplier for the final two layers of the MLP portion of the final transformer. (Default: 1)

Once instantiated, an `OmLLM` object can be called as follows:

```python
output, state_token_sequence_end, offset = omllm(input_sequence, state_token_sequence, offset, next_token_flag)
```

The `output` `Tensor` will contain logits for each next-token prediction. If `next_token_flag` is `False` (the default) have dimensions `(batch_size, seq_len, vocab_size_adj)`, where `vocab_size_adj` is equal to the input vocab size plus an offset to make the number divisible by 8 (which is done to make downstream softmax computations more efficient on CUDA devices -- note, all "padded" logits are set to `-inf`); if `next_token_flag` is `True`, the `Tensor` will have dimensions `(batch_size, vocab_size_adj)` and represent only the logits for the final next-token in the input.

The `state_token_sequence_end` `Tensor` will have the same shape as `state_token_sequence`, and can be stored to be used as the initial state token sequence for the next call to the model (context caching).

The `offset` `int` will be the offset for the input sequence. This can be used to perform context caching.

## Installation

To install Om LLM, run the following commands:

```bash
git clone https://github.com/dingo-actual/om.git
cd om
pip install -r requirements.txt
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- xformers 0.0.26+
- NumPy 1.25+

## Future Work

- Add a linear projection after the final memory layer in `ARC` to bring the output dimension of the memory block to the same dimension as the first memory layer.
  - This will allow for the final memory dimension to be set arbitrarily, instead of being fixed at the dimension of the first memory layer.
- Investigate the properties of the `ARCformer` initial state token sequence after training.
  - Hopefully, this allows for the initial state token sequence to be initialized in a static fashion, rather than being learned (which may make training more difficult).
- Investigate the impact of the state token sequence length on the length generalization of the model.

## Contributing

We welcome contributions to Om LLM. Please see our [Contributing Guidelines](CONTRIBUTING.md) for more information.

## License

Om LLM is released under the [Apache 2.0 License](LICENSE).
