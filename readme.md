# Om LLM

<img src="om_llm.jpg" alt="Om LLM Logo" width="25%" height="25%">
<br/>
<br/>

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

### `ARC` Usage

## `ARCformer`: Attentive Recurrent Cell (ARC) Transformer

The ARC Transformer (`ARCformer`) is the second highest-level component of Om LLM. It represents a transformer-like architecture that incorporates multi-pass memory, as well as "state token sequences". The forward pass of `ARCformer` requires inputs of a sub-sequence and a current state token sequence (the initial state for each `ARCformer` is a learned parameter) and will produce two outputs:

- The usual output sequence from a transformer block, as well as
- A state token sequence, which is used to process the next input sub-sequence.

### `ARCformer` Usage

## `RoPEEmbeddings`

### `RoPEEmbeddings` Usage

## `OmLLM`

### `OmLLM` Usage

## Installation

## Requirements

## Contributing

We welcome contributions to Om LLM. Please see our [Contributing Guidelines](CONTRIBUTING.md) for more information.

## License

Om LLM is released under the [Apache 2.0 License](LICENSE).
