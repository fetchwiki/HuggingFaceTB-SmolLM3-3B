# HuggingFaceTB/SmolLM3-3B

HuggingFace model: HuggingFaceTB/SmolLM3-3B

This is a mirror of the [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) repository from HuggingFace.

**Note**: This repository contains metadata and configuration files only. The actual model files are stored on HuggingFace due to their large size.

## Model Information

### Architecture

- **Architecture**: SmolLM3ForCausalLM
- **Model Type**: smollm3
- **Vocabulary Size**: 128,256
- **Hidden Size**: 2,048
- **Number of Layers**: 36
- **Attention Heads**: 16

### Model Files

This repository contains metadata about the model files. The actual model files are stored on HuggingFace.

| File | Size | Details |
|------|------|---------|
| [model-00001-of-00002.safetensors](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/model-00001-of-00002.safetensors) | 4GB | 258 tensors |
| [model-00002-of-00002.safetensors](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/model-00002-of-00002.safetensors) | 1GB | 68 tensors |

**Total files**: 2

### Tensor Details

Sample tensors from safetensors files:

**model-00001-of-00002.safetensors**:
    - model.embed_tokens.weight: shape=[128256,2048], dtype=BF16
    - model.layers.0.input_layernorm.weight: shape=[2048], dtype=BF16
    - model.layers.0.mlp.down_proj.weight: shape=[2048,11008], dtype=BF16
    - model.layers.0.mlp.gate_proj.weight: shape=[11008,2048], dtype=BF16
    - model.layers.0.mlp.up_proj.weight: shape=[11008,2048], dtype=BF16

**model-00002-of-00002.safetensors**:
    - model.layers.28.input_layernorm.weight: shape=[2048], dtype=BF16
    - model.layers.28.mlp.down_proj.weight: shape=[2048,11008], dtype=BF16
    - model.layers.28.mlp.up_proj.weight: shape=[11008,2048], dtype=BF16
    - model.layers.28.post_attention_layernorm.weight: shape=[2048], dtype=BF16
    - model.layers.29.input_layernorm.weight: shape=[2048], dtype=BF16

## Usage

To use this model, visit the original HuggingFace repository:
- [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)

## Additional Information

This mirror was created to provide easy access to model metadata and configuration files. For the actual model weights and full functionality, please visit the original repository on HuggingFace.

**Generated**: 2025-07-14
