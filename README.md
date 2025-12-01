# HOPE - Hierarchical Online Predictive Encoding

Implementation of the HOPE architecture based on the [Nested Learning: The Illusion of Deep Learning
Architectures](https://abehrouz.github.io/files/NL.pdf)

## Architecture

HOPE combines:

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
from hope.config import HopeSmallConfig
from hope.model import HopeForCausalLM

config = HopeSmallConfig(vocab_size=32000)
model = HopeForCausalLM(config)

# Forward pass
outputs = model(input_ids=input_ids, labels=labels)
loss = outputs["loss"]
```

### Memory Management

```python
from hope.model import Hope

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

| Size  | Parameters |
| ----- | ---------- |
| Small | 68M        |
| Base  | 279M       |
| Large | 978M       |

## Training

```bash
uv run python train.py --model_size small --batch_size 8 --learning_rate 1e-4
```

Options:

- `--optimizer`: adamw, adam_delta, sgd_delta, deep_momentum
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
hope/
    config.py           # Model configurations
    model.py            # Main Hope model
    modules/
        titans.py       # Self-Modifying Titans
        continuum_memory.py  # CMS
        hope_block.py   # Combined block
    layers/
        associative_memory.py  # Delta rule memory
        neural_memory.py       # Neural memory modules
    optimizers.py       # Custom optimizers (Adam+Delta, Muon, etc.)
```

## Reference

[Nested Learning - The Illusion of Deep Learning Architectures](https://abehrouz.github.io/files/NL.pdf)
