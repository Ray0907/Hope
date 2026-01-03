# HOPE - Hierarchical Optimization with Persistent Experience

Implementation of the HOPE architecture based on:

- [Nested Learning: The Illusion of Deep Learning Architectures](https://abehrouz.github.io/files/NL.pdf)
- [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)
- [MIRAS: Memory Is All You Need](https://arxiv.org/abs/2504.13173)

## Architecture Overview

HOPE combines core components from the Nested Learning and Titans papers, plus the MIRAS unified framework:

- **Self-Modifying Titans**: Memory attention with delta rule updates (Eq. 28-29)
- **Continuum Memory System (CMS)**: Multi-frequency FFN chain (Eq. 30-31)
- **MIRAS Framework**: Unified memory system with configurable attentional bias and retention gates

### Delta Rule (Eq. 28-29)

```
M_{t+1} = M_t - M_t * k_t * k_t^T - eta * (M_t * k_t - v_t) * k_t^T
```

Where:

- First term: Forgetting (removes old association for key)
- Second term: Learning (gradient descent on L2 loss)

### Titans Variants

Three architectural variants for integrating memory with attention:

| Variant | Config | Description |
|---------|--------|-------------|
| **MAC** | `mac` | Memory as Context - memory output concatenated with attention (default) |
| **MAG** | `mag` | Memory as Gate - memory gates attention output via sigmoid |
| **MAL** | `mal` | Memory as Layer - memory pre-processes input before attention |

```python
# Select variant via config
config = HopeSmallConfig(titans_variant="mag")  # mac, mag, mal

# Or use directly
from src.modules import MemoryAsGate, MemoryAsLayer

mag = MemoryAsGate(dim=512, num_heads=8)
mal = MemoryAsLayer(dim=512, num_heads=8)
```

**MAG (Memory as Gate)**:
```
attn_out = softmax(QK^T/sqrt(d)) @ V
gate = sigmoid(Memory @ q)
output = gate * attn_out
```

**MAL (Memory as Layer)**:
```
enriched = x + Memory @ x
output = Attention(enriched)
```

### MIRAS Framework

The MIRAS framework unifies sequence models through 4 design choices:

| Choice | Options | Description |
|--------|---------|-------------|
| Memory Architecture | Vector, Matrix, MLP | How memory is structured |
| Attentional Bias | L2, Lp, Huber, KL | Internal memory objective |
| Retention Gate | L2, Lq, KL, Elastic Net | How to retain past state |
| Learning Algorithm | GD, GD+Momentum, Newton | How to update memory |

Three pre-configured MIRAS models:

| Model | Attentional Bias | Retention Gate | Use Case |
|-------|------------------|----------------|----------|
| **Moneta** | Lp (p in (1,2)) | Lq (q in (1,2)) | Robust to key collisions |
| **Yaad** | Huber Loss | L2 | Robust to outlier values |
| **Memora** | L2 | KL-divergence | Soft thresholding |

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

### MIRAS Models

```python
from src.layers import Moneta, Yaad, Memora, MirasMemory

# Pre-configured models
moneta = Moneta(dim=512, num_heads=8, p=1.5, q=1.5)
yaad = Yaad(dim=512, num_heads=8, huber_delta=1.0)
memora = Memora(dim=512, num_heads=8, kl_temperature=1.0)

# Custom configuration
memory = MirasMemory(
    dim_key=64, dim_value=64,
    attentional_bias="huber",  # l2, lp, huber, kl, dot_product
    retention_gate="elastic_net",  # l2, lq, kl, elastic_net, bregman
    learning_rate=0.1,
    retention_strength=0.1,
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
        titans.py          # Self-Modifying Titans (MAC, MAG, MAL)
        continuum_memory.py  # CMS and variants
        hope_block.py      # Combined HOPE block
    layers/
        __init__.py
        associative_memory.py  # Delta rule memory
        neural_memory.py       # MLP-based neural memory
        attentional_bias.py    # MIRAS attentional bias (L2, Lp, Huber, KL)
        retention_gates.py     # MIRAS retention gates (L2, Lq, KL, Elastic Net)
        miras_memory.py        # MIRAS models (Moneta, Yaad, Memora)
```

## Reference

- [Nested Learning - The Illusion of Deep Learning Architectures](https://abehrouz.github.io/files/NL.pdf)
- [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)
- [MIRAS: Memory Is All You Need](https://arxiv.org/abs/2504.13173)

## License

MIT License
