# HOPE - Hierarchical Optimization with Persistent Experience

Implementation of the HOPE architecture based on the [Nested Learning: The Illusion of Deep Learning Architectures](https://abehrouz.github.io/files/NL.pdf) paper.

## Architecture Overview

HOPE combines two core components from the Nested Learning paper:

- **Self-Modifying Titans**: Memory attention with delta rule updates (Eq. 28-29)
- **Continuum Memory System (CMS)**: Multi-frequency FFN chain (Eq. 30-31)

### Delta Rule (Eq. 28-29)

```
M_{t+1} = M_t - M_t * k_t * k_t^T - eta * (M_t * k_t - v_t) * k_t^T
```

Where:

- First term: Forgetting (removes old association for key)
- Second term: Learning (gradient descent on L2 loss)

## Installation

Using uv (recommended):

```bash
uv sync
```

Or using pip:

```bash
pip install torch
```

## Usage

### Basic Usage

```python
from src.config import HopeSmallConfig
from src.model import HopeForCausalLM

config = HopeSmallConfig(vocab_size=32000)
model = HopeForCausalLM(config)

# Forward pass
outputs = model(input_ids=input_ids, labels=labels)
loss = outputs["loss"]
```

### Memory Management

```python
from src.model import Hope

model = Hope(config)
memory_states = None

for batch in dataloader:
    logits, memory_states = model(
        batch["input_ids"],
        memory_states=memory_states,
        return_memory=True,
    )
```

### Text Generation

```python
generated = model.generate(
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9,
)
```

## Model Sizes

| Size  | Parameters | dim  | layers | heads |
| ----- | ---------- | ---- | ------ | ----- |
| Small | ~125M      | 512  | 8      | 8     |
| Base  | ~350M      | 768  | 12     | 12    |
| Large | ~760M      | 1024 | 24     | 16    |
| XL    | ~1.3B      | 2048 | 24     | 32    |

## Training

```bash
uv run python train.py --model_size small --batch_size 8 --learning_rate 1e-4
```

Options:

- `--optimizer`: adamw, adam_delta, sgd_delta, deep_momentum, muon
- `--lr_scheduler`: cosine, linear, constant
- `--dtype`: float32, float16, bfloat16

## Testing

```bash
uv run python test_hope.py
```

## Examples

```bash
uv run python example.py
```

## Project Structure

```
src/
    __init__.py
    config.py              # Model configurations
    model.py               # Main Hope model
    optimizers.py          # Deep optimizers (DMGD, Muon, etc.)
    modules/
        __init__.py
        titans.py          # Self-Modifying Titans variants
        continuum_memory.py  # CMS and variants
        hope_block.py      # Combined HOPE block
    layers/
        __init__.py
        associative_memory.py  # Delta rule memory
        neural_memory.py       # MLP-based neural memory
```

## Reference

- [Nested Learning - The Illusion of Deep Learning Architectures](https://abehrouz.github.io/files/NL.pdf)
- [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)

## License

MIT License
