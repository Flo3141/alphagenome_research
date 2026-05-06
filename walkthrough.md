# RNA Half-Life Fine-tuning – Walkthrough

## Overview

Four files were modified/extended in `src/alphagenome_research/` to support predicting RNA half-life as a scalar regression task on top of the frozen AlphaGenome trunk.

---

## Files Changed

### 1. `model/schemas.py` — New `rna_half_life` field

Added one field to `DataBatch`:

```python
rna_half_life: Float[ArrayLike, 'B'] | None = None
```

This carries the per-sequence scalar target (half-life in natural units, e.g. minutes).

---

### 2. `model/heads.py` — `RNAHalfLifeHead` + config wiring

**Enum extensions:**
- `HeadType.RNA_HALF_LIFE = 'rna_half_life'`
- `HeadName.RNA_HALF_LIFE = 'rna_half_life'`

**New config dataclass:**
```python
@dataclasses.dataclass
class RNAHalfLifeHeadConfig(HeadConfig):
  hidden_dim: int = 512
  dropout_rate: float = 0.1
  huber_delta: float = 1.0
```

**Factory wiring:** `create_head()` and `get_head_config()` dispatch on the new enum values.

**`RNAHalfLifeHead` class** (does **not** inherit from `Head` — no multi-organism metadata needed):

| Step | Operation |
|------|-----------|
| 1 | Global mean-pool of `embeddings_128bp` → `[B, 3072]` |
| 2 | `Linear(3072→512) + GELU + optional Dropout` |
| 3 | `Linear(512→256) + GELU` |
| 4 | `Linear(256→1)` → squeeze → `[B]` |

**Loss:** Huber (smooth L1) in **log1p space** — targets are transformed as `log1p(half_life)` before comparison to predictions. Returns `{'loss': scalar, 'mae': scalar}`.

---

### 3. `model/model.py` — Integration into `AlphaGenome`

- `__init__`: Initialises `_rna_half_life_head = None` and skips `RNA_HALF_LIFE` in the main head-discovery loop (no track metadata).
- `enable_rna_half_life_head(hidden_dim, dropout_rate, huber_delta)`: Opt-in method that creates the head object. Call **before** `hk.transform_with_state`.
- `__call__`: Accepts `is_training: bool = False` for dropout; calls the half-life head when enabled.
- `loss`: Includes half-life Huber loss (weighted by `loss_weight`).

---

### 4. `finetuning/dataset.py` — `RNAHalfLifeDataPipeline`

A CSV-based pipeline (no BigWig required).

**Required CSV columns:** `chromosome`, `start`, `end`, `half_life`

```python
pipeline = RNAHalfLifeDataPipeline(
    fasta_path='path/to/hg38.fa',
    labels_csv_path='path/to/labels.csv',
    sequence_length=524_288,
)
```

Yields `{'dna_sequence': [S,4], 'organism_index': 0, 'rna_half_life': float32}`.

---

### 5. `finetuning/finetune.py` — Trunk-freezing training loop

**New API functions:**

| Function | Purpose |
|----------|---------|
| `get_rna_half_life_dataset_iterator(...)` | Wraps `RNAHalfLifeDataPipeline` into a batched `DataBatch` iterator |
| `get_rna_half_life_forward_fn(...)` | Haiku `transform_with_state` with trunk frozen via `stop_gradient` |
| `get_rna_half_life_train_step(forward_fn, optimizer)` | Jitted step with dual-level trunk freezing |
| `init_rna_half_life_training(...)` | Convenience: init params, load checkpoint, return `(params, state, opt_state, train_step)` |

#### Dual-Level Trunk Freezing

```
Level 1 — stop_gradient (in forward pass):
  AlphaGenome(freeze_trunk_embeddings=True)
  → jax.lax.stop_gradient(embeddings)
  → trunk receives zero gradients from jax.grad

Level 2 — optax.masked (in optimiser):
  optax.masked(
      {True: base_optimizer, False: optax.set_to_zero()},
      mask,   # True = head params, False = trunk params
  )
  → trunk parameter deltas are explicitly zeroed
  → trunk weights are guaranteed unchanged
```

---

## Minimal Usage Example

```python
import optax
from alphagenome.models import dna_model
from alphagenome_research.finetuning import finetune
from alphagenome_research.model.metadata import metadata as metadata_lib

# 1. Load metadata (needed to initialise trunk heads)
output_metadata = {
    dna_model.Organism.HOMO_SAPIENS: metadata_lib.load(
        dna_model.Organism.HOMO_SAPIENS
    )
}

# 2. Build the forward function (trunk frozen, half-life head enabled)
forward_fn = finetune.get_rna_half_life_forward_fn(output_metadata)

# 3. Build the dataset iterator
data_iter = finetune.get_rna_half_life_dataset_iterator(
    batch_size=4,
    sequence_length=524_288,
    labels_csv_path='data/half_life_labels.csv',
)

# 4. Initialise training (pass pretrained_params from a checkpoint)
params, state, opt_state, train_step = finetune.init_rna_half_life_training(
    forward_fn=forward_fn,
    base_optimizer=optax.adam(1e-4),
    sample_batch=next(data_iter),
    pretrained_params=pretrained_params,   # from your checkpoint
    pretrained_state=pretrained_state,
)

# 5. Training loop
for batch in data_iter:
    params, state, opt_state, scalars = train_step(
        params, state, opt_state, batch
    )
    print(scalars['loss'], scalars['rna_half_life_mae'])
```

> [!IMPORTANT]
> The `labels.csv` must contain columns: `chromosome`, `start`, `end`, `half_life`.
> The `half_life` values should be in natural units (e.g. minutes) — **not** pre-log-transformed.

> [!NOTE]
> The original `get_forward_fn` / `get_train_step` API is **unchanged**. All existing code continues to work without modification.
