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

To the best of my knowledge, these features are novel. If you find any references to these features in the literature, please let me know!

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

The `output` tensor will have the same shape as the input sequence, and can be passed to the MLP portion of an `ARCformer`. The `state_token_sequence` tensor will have the same shape as `state_token_sequence`, and can be passed as the `state=` argument for the next sub-sequence processed by this `ARC` layer.

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
- dropout (`float`): The dropout rate for the MLP portion of the transformer. (Default: 0.0)
- mlp_multiplier (`int`): Multiplier for the final two layers of the MLP portion of the transformer. (Default: 1)

Once instantiated, an `ARCformer` object can be called as follows:

```python
output, state_token_sequence = arcformer(input_sequence, state_token_sequence, offset)
```

The `output` tensor will have the same shape as the input sequence, and can be passed to the next `ARCformer` in the model. The `state_token_sequence` tensor will have the same shape as `state_token_sequence`, and can be passed as the `state=` argument for the next sub-sequence processed by this `ARCformer`.

## `RoPEEmbeddings`

RoPEEmbeddings is a class that implements the RoPE (Rotary Position Embedding) positional embedding scheme, as described in the paper "RoFormer: Enhanced Transformer with Rotary Position Embedding" by Jianlin Su et al. [arxiv](https://arxiv.org/abs/2104.09864). It has minor modifications made to support `ARC`'s recurrent structure.

### `RoPEEmbeddings` Usage

TODO

## `OmLLM`

TODO

### `OmLLM` Usage

TODO

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
