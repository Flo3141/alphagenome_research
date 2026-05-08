# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AlphaGenome finetuning script.

This module provides two independent fine-tuning workflows:

1. **Original workflow** (``get_forward_fn`` / ``get_train_step``): Fine-tunes
   all heads jointly using the pre-trained trunk.  All parameters receive
   gradient updates.  Trunk embeddings can optionally be frozen via
   ``stop_gradient`` (``freeze_trunk_embeddings=True`` in ``AlphaGenome``).

2. **RNA half-life workflow** (``get_rna_half_life_forward_fn`` /
   ``get_rna_half_life_train_step``): Fine-tunes *only* the new
   ``RNAHalfLifeHead`` while keeping the AlphaGenome trunk weights fixed.
   Parameter freezing is implemented at two levels:
     a. **stop_gradient** on the trunk embeddings (via
        ``freeze_trunk_embeddings=True``) prevents gradients from flowing
        back through the trunk computation.
     b. **optax.masked** zeroes out any residual gradient updates to trunk
        parameters so they can never be modified by the optimiser, even if
        gradients accidentally leak through.
   This dual strategy is the recommended approach in the JAX/Haiku ecosystem
   when the pre-trained trunk must remain completely unchanged.
"""

from collections.abc import Iterator, Mapping
from typing import Any, Callable

from alphagenome.data import fold_intervals
from alphagenome.models import dna_model
from alphagenome_research.finetuning import dataset as dataset_lib
from alphagenome_research.model import heads as heads_module
from alphagenome_research.model import model
from alphagenome_research.model import schemas
from alphagenome_research.model.metadata import metadata as metadata_lib
import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import optax
import tensorflow as tf

_FASTA_PATH = (
    'https://storage.googleapis.com/alphagenome/reference/gencode/'
    'hg38/GRCh38.p13.genome.fa'
)


# =============================================================================
# Original fine-tuning API  (unchanged from upstream)
# =============================================================================


def get_dataset_iterator(
    *,
    batch_size: int,
    sequence_length: int,
    output_metadata: metadata_lib.AlphaGenomeOutputMetadata,
    model_version: dna_model.ModelVersion,
    subset: fold_intervals.Subset,
    organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
    fasta_path: str = _FASTA_PATH,
    example_regions_path: str | None = None,
) -> Iterator[schemas.DataBatch]:
  """Converts pipeline output dict to a DataBatch schema.

  Args:
    batch_size: The batch size of the dataset.
    sequence_length: The sequence length of the dataset.
    output_metadata: Metadata for the output tracks for each organism.
    model_version: The model version to use.
    subset: The subset of the dataset.
    organism: The organism to use.
    fasta_path: The path to the reference genome FASTA file.
    example_regions_path: The path to the example regions BED file.

  Returns:
    A schemas.DataBatch object.
  """
  intervals = fold_intervals.get_fold_intervals(
      model_version,
      organism,
      subset,
      example_regions_path=example_regions_path,
  )
  pipeline = dataset_lib.DataPipeline(
      fasta_path=fasta_path,
      intervals=intervals,
      output_metadata=output_metadata,
      organism=organism,
      sequence_length=sequence_length,
  )
  dataset = tf.data.Dataset.from_generator(
      pipeline.get_generator,
      output_signature=pipeline.get_output_signature(),
  )
  dataset = (
      dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
  )

  def iterator():
    for batch in dataset:
      yield schemas.DataBatch(
          dna_sequence=batch['dna_sequence'],
          organism_index=batch['organism_index'],
          **batch['bundles'],
      )

  return iterator()


def get_forward_fn(
    output_metadata: Mapping[
        dna_model.Organism, metadata_lib.AlphaGenomeOutputMetadata
    ],
    jmp_policy: str = 'params=float32,compute=bfloat16,output=bfloat16',
) -> hk.TransformedWithState:
  """Creates a Haiku transformed function for the AlphaGenome model.

  Args:
    output_metadata: Metadata for the output tracks for each organism.
    jmp_policy: The JMP policy to use for mixed precision.

  Returns:
    A `hk.TransformedWithState` object representing the forward pass.
  """
  jmp_policy = jmp.get_policy(jmp_policy)

  @hk.transform_with_state
  def forward(batch: schemas.DataBatch):
    with hk.mixed_precision.push_policy(model.AlphaGenome, jmp_policy):
      return model.AlphaGenome(
          output_metadata, freeze_trunk_embeddings=True
      ).loss(batch)

  return forward


def get_train_step(
    predict_fn: Callable[..., Any],
    optimizer: optax.GradientTransformation,
):
  """Creates a jitted training step function using AlphaGenome.loss.

  Args:
    predict_fn: The Haiku transformed forward function.
    optimizer: An Optax optimizer to apply gradients.

  Returns:
    A jitted function `train_step` that takes `params`, `state`, `opt_state`,
    and a `batch` as input and returns the updated `params`, `next_state`,
    `new_opt_state`, and a dictionary of `metrics`.
  """

  @jax.jit
  def train_step(params, state, opt_state, batch):
    def loss_fn(params, state, batch):
      (loss, scalars, predictions), new_state = predict_fn(
          params, state, None, batch
      )
      del predictions  # Unused.
      return loss, (new_state, scalars)

    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (next_state, scalars)), grads = loss_grad_fn(params, state, batch)
    scalars['loss'] = loss
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, next_state, new_opt_state, scalars

  return train_step


# =============================================================================
# RNA Half-Life Fine-tuning API
# =============================================================================


def get_rna_half_life_dataset_iterator(
    *,
    batch_size: int,
    sequence_length: int,
    labels_csv_path: str,
    fasta_path: str = _FASTA_PATH,
    organism: dna_model.Organism = dna_model.Organism.HOMO_SAPIENS,
    shuffle: bool = True,
    seed: int = 0,
    sep: str = ',',
) -> Iterator[schemas.DataBatch]:
  """Returns a batched DataBatch iterator for RNA half-life regression.

  Reads genomic intervals and scalar half-life labels from a CSV file and
  yields batched ``schemas.DataBatch`` objects ready for use with
  ``get_rna_half_life_train_step``.

  The CSV must contain columns: ``chromosome``, ``start``, ``end``,
  ``half_life`` (see ``RNAHalfLifeDataPipeline`` for details).

  Args:
    batch_size: Number of sequences per batch.
    sequence_length: Fixed sequence length (typically 524_288 bp).
    labels_csv_path: Path to the CSV file containing genomic intervals and
      half-life labels.
    fasta_path: Path or URL to the reference genome FASTA file.
    organism: The organism.  Currently only HOMO_SAPIENS is supported.
    shuffle: Whether to shuffle examples at the start of each epoch.
    seed: Random seed for reproducibility.
    sep: Column separator for the CSV file.

  Returns:
    An iterator that yields ``schemas.DataBatch`` objects with
    ``dna_sequence``, ``organism_index``, and ``rna_half_life`` fields.
  """
  pipeline = dataset_lib.RNAHalfLifeDataPipeline(
      fasta_path=fasta_path,
      labels_csv_path=labels_csv_path,
      sequence_length=sequence_length,
      organism=organism,
      sep=sep,
  )

  dataset = tf.data.Dataset.from_generator(
      lambda: pipeline.get_generator(shuffle=shuffle, seed=seed),
      output_signature=pipeline.get_output_signature(),
  )
  dataset = (
      dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
  )

  def iterator():
    for batch in dataset:
      yield schemas.DataBatch(
          dna_sequence=batch['dna_sequence'],
          organism_index=batch['organism_index'],
          # The scalar half-life label is passed through the rna_half_life
          # field of DataBatch, which was added specifically for this task.
          rna_half_life=batch['rna_half_life'],
      )

  return iterator()


def get_rna_half_life_forward_fn(
    output_metadata: Mapping[
        dna_model.Organism, metadata_lib.AlphaGenomeOutputMetadata
    ],
    *,
    hidden_dim: int = 512,
    dropout_rate: float = 0.1,
    huber_delta: float = 1.0,
    jmp_policy: str = 'params=float32,compute=bfloat16,output=bfloat16',
) -> hk.TransformedWithState:
  """Creates the Haiku forward function for RNA half-life fine-tuning.

  The returned function runs the full AlphaGenome trunk (encoder + transformer
  + decoder) with ``freeze_trunk_embeddings=True``, which wraps the trunk
  output in ``jax.lax.stop_gradient``.  The ``RNAHalfLifeHead`` is then
  attached on top.

  Args:
    output_metadata: Organism-level track metadata (same as for the
      pre-trained model).  This is still required to initialise the trunk
      heads correctly so that pre-trained weights can be loaded.
    hidden_dim: MLP hidden dimension for the RNA half-life head.
    dropout_rate: Dropout rate inside the RNA half-life head.
    huber_delta: Huber loss delta for the RNA half-life head loss.
    jmp_policy: JMP mixed-precision policy string.

  Returns:
    A ``hk.TransformedWithState`` whose ``apply`` signature is:
    ``(params, state, rng, batch: DataBatch, is_training: bool)``
    and returns ``(loss, scalars, predictions)``.
  """
  jmp_policy_obj = jmp.get_policy(jmp_policy)

  @hk.transform_with_state
  def forward(batch: schemas.DataBatch, is_training: bool = False):
    with hk.mixed_precision.push_policy(model.AlphaGenome, jmp_policy_obj):
      # Build the model with trunk-freezing enabled.
      #
      # ``freeze_trunk_embeddings=True`` inserts a ``stop_gradient`` call on
      # the trunk embeddings *before* they are passed to any head.  This
      # means:
      #   - No gradient flows back into the trunk layers (SequenceEncoder,
      #     TransformerTower, SequenceDecoder, OutputEmbedder) during the
      #     backward pass of the RNA half-life loss.
      #   - The trunk parameters will receive a zero gradient tensor, so the
      #     optimiser sees them but computes zero updates.
      #
      # This ``stop_gradient`` layer is the *primary* freezing mechanism and
      # is sufficient on its own.  The ``optax.masked`` strategy applied in
      # ``get_rna_half_life_train_step`` provides a *secondary* safety net
      # that explicitly zeroes out any trunk parameter updates at the
      # optimiser level.
      alpha_genome_model = model.AlphaGenome(
          output_metadata, freeze_trunk_embeddings=True
      )
      # Attach the RNA half-life head.  This must be called before the first
      # forward pass so that Haiku creates the head's parameters on init.
      alpha_genome_model.enable_rna_half_life_head(
          hidden_dim=hidden_dim,
          dropout_rate=dropout_rate,
          huber_delta=huber_delta,
      )
      return alpha_genome_model.loss(batch, is_training=is_training)

  return forward


def _make_trunk_mask(params: Any) -> Any:
  """Creates a boolean pytree mask identifying trunk vs. head parameters.

  Haiku stores parameters in a nested dict keyed by module name (the first
  level).  After calling ``enable_rna_half_life_head``, the RNA half-life
  head's parameters live under the key ``'alphagenome/head/rna_half_life'``
  (or similar, depending on the name_scope nesting).

  This function walks the top-level parameter dict and returns ``True`` for
  every leaf that belongs to the RNA half-life head and ``False`` for all
  other (trunk) leaves.

  The returned mask is used with ``optax.masked`` so that:
  - ``True``  → the head's parameters ARE updated by the optimiser.
  - ``False`` → the trunk parameters are FROZEN (the optimiser returns
                zero-valued updates for them).

  Args:
    params: A Haiku parameter pytree (nested dict of arrays) as returned by
      ``hk.transform_with_state(...).init``.

  Returns:
    A pytree of the same structure as ``params``, where leaves are booleans:
    ``True`` if the parameter belongs to the RNA half-life head, ``False``
    otherwise.

  Note:
    The module name prefix used here (``'rna_half_life'``) must match the
    ``name`` argument passed to ``RNAHalfLifeHead.__init__``.  The default
    value is ``'rna_half_life'``.  If you change the head name, update this
    function accordingly.
  """
  # The ``_HEAD_MODULE_PREFIX`` is a substring that uniquely identifies the
  # RNA half-life head's parameter scope.  Haiku joins scopes with '/', so
  # the full key will look like 'alphagenome/head/rna_half_life'.
  _HEAD_MODULE_PREFIX = 'rna_half_life'

  def _is_head_param(path, _leaf):
    """Returns True if the path (tuple of dict keys) points to a head param."""
    # jax.tree_util.tree_map_with_path provides the path as a sequence of
    # jax.tree_util.DictKey objects.  We extract the string key from each.
    for key_entry in path:
      key_str = (
          key_entry.key
          if hasattr(key_entry, 'key')
          else str(key_entry)
      )
      if _HEAD_MODULE_PREFIX in key_str:
        return True
    return False

  # jax.tree_util.tree_map_with_path walks the pytree and provides both the
  # path (sequence of keys) and the leaf value.  We only care about the path.
  mask = jax.tree_util.tree_map_with_path(
      lambda path, leaf: _is_head_param(path, leaf),
      params,
  )
  return mask


def get_rna_half_life_train_step(
    forward_fn: hk.TransformedWithState,
    base_optimizer: optax.GradientTransformation,
) -> Callable[..., tuple[Any, Any, Any, dict[str, Any]]]:
  """Creates a jitted training step that freezes the trunk.

  Parameter freezing strategy
  ---------------------------
  Freezing is implemented at **two independent levels** for robustness:

  **Level 1 — stop_gradient on embeddings** (in the forward pass):
    ``AlphaGenome(freeze_trunk_embeddings=True)`` wraps the trunk output
    in ``jax.lax.stop_gradient``.  This prevents any gradient from flowing
    backwards through the trunk layers during ``jax.grad``, so trunk
    parameters receive zero gradients.

  **Level 2 — optax.masked on the optimiser** (in the train step):
    ``optax.masked(base_optimizer, mask)`` applies the optimiser *only* to
    parameters where ``mask`` is ``True`` (i.e., the RNA half-life head).
    For parameters where ``mask`` is ``False`` (i.e., the trunk), the
    optimiser applies ``optax.set_to_zero()``, which explicitly zeroes out
    any update — ensuring the trunk weights can never change, even if a
    gradient accidentally leaks through.

  The ``mask`` pytree is computed once on the *first call* (after JAX has
  traced the function) and is then fixed for the lifetime of the training
  loop.  This is safe because the parameter structure does not change during
  training.

  Args:
    forward_fn: The Haiku transformed function returned by
      ``get_rna_half_life_forward_fn``.
    base_optimizer: An Optax gradient transformation to apply to the head
      parameters.  Common choices: ``optax.adam(lr)``,
      ``optax.adamw(lr, weight_decay=...)``.

  Returns:
    A jitted ``train_step`` function with signature::

      train_step(params, state, opt_state, batch)
          -> (new_params, new_state, new_opt_state, scalars)

    where ``scalars`` is a dict containing at minimum ``'loss'`` and
    ``'rna_half_life_mae'``.
  """

  # ``opt_state`` is initialised lazily inside ``train_step`` so that we can
  # compute the mask from the actual parameter structure.
  #
  # We store the masked optimiser in a mutable container so the JIT-compiled
  # function can reference it after the first call resolves the mask.
  _optimizer_cache: dict[str, optax.GradientTransformation] = {}

  def _get_masked_optimizer(params):
    """Builds (or retrieves) the masked optimiser for the given params."""
    if 'optimizer' not in _optimizer_cache:
      # Build the boolean mask pytree: True → head param, False → trunk param.
      mask = _make_trunk_mask(params)

      # ``optax.masked`` applies ``base_optimizer`` where ``mask`` is True,
      # and ``optax.set_to_zero()`` where ``mask`` is False.
      #
      # optax.set_to_zero() produces zero-valued updates, which means:
      #   new_param = old_param + 0  =>  param is unchanged.
      #
      # This is the *secondary* freezing mechanism (the primary is
      # ``stop_gradient``).  Using both gives defence-in-depth: even if a
      # future code change accidentally removes the ``stop_gradient``, the
      # optimiser will still not update the trunk parameters.
      masked_optimizer = optax.multi_transform(
          # Inner dict: maps mask value to the transform to apply.
          # True  → apply the real optimiser (update head params).
          # False → apply set_to_zero (freeze trunk params).
          {True: base_optimizer, False: optax.set_to_zero()},
          mask,
      )
      _optimizer_cache['optimizer'] = masked_optimizer
    return _optimizer_cache['optimizer']

  @jax.jit
  def train_step(
      params: Any,
      state: Any,
      opt_state: Any,
      rng: hk.PRNGKey,
      batch: schemas.DataBatch,
  ) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Performs one gradient update step for the RNA half-life head only.

    Args:
      params: Current model parameters (trunk + head).
      state: Current Haiku model state (e.g., BatchNorm statistics).
      opt_state: Current optimiser state.
      batch: A ``DataBatch`` with ``dna_sequence``, ``organism_index``, and
        ``rna_half_life`` fields.

    Returns:
      A 4-tuple ``(new_params, new_state, new_opt_state, scalars)`` where:
      - ``new_params``: Updated parameters.  **Trunk params are unchanged.**
      - ``new_state``: Updated Haiku state (e.g., updated BatchNorm stats).
      - ``new_opt_state``: Updated optimiser state.
      - ``scalars``: Dict of monitoring metrics including ``'loss'``.
    """

    def loss_fn(params, state, rng, batch):
      # ``is_training=True`` enables dropout inside the head.
      (loss, scalars, predictions), new_state = forward_fn.apply(
          params, state, rng, batch, True  # is_training=True
      )
      del predictions  # Unused.
      return loss, (new_state, scalars)

    # Compute loss value and gradients w.r.t. all parameters.
    # Because of ``stop_gradient`` on trunk embeddings, the trunk gradients
    # will be zero tensors — they are computed by JAX but carry no signal.
    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (next_state, scalars)), grads = loss_grad_fn(params, state, rng, batch)
    scalars['loss'] = loss

    # Retrieve the masked optimiser (built on first call).
    # The mask zeroes out trunk parameter updates at the optimiser level.
    # This is the SECONDARY freezing mechanism complementing stop_gradient.
    masked_optimizer = _get_masked_optimizer(params)

    # Apply the masked optimiser:
    #   - Head parameters: updated by base_optimizer using their (non-zero) grads.
    #   - Trunk parameters: replaced with zero updates by set_to_zero(),
    #     so ``new_params`` trunk == ``params`` trunk.
    updates, new_opt_state = masked_optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, next_state, new_opt_state, scalars

  return train_step


def init_rna_half_life_training(
    forward_fn: hk.TransformedWithState,
    base_optimizer: optax.GradientTransformation,
    sample_batch: schemas.DataBatch,
    pretrained_params: Any | None = None,
    pretrained_state: Any | None = None,
) -> tuple[Any, Any, Any, Callable[..., Any]]:
  """Initialises all components needed for RNA half-life fine-tuning.

  This is a convenience function that wires together parameter initialisation,
  pre-trained weight loading, optimiser state initialisation, and the
  masked training step.

  Args:
    forward_fn: The result of ``get_rna_half_life_forward_fn``.
    base_optimizer: The Optax optimiser to use for the head parameters.
    sample_batch: A single representative batch used to initialise model
      parameters.  The batch must contain all required fields
      (``dna_sequence``, ``organism_index``, ``rna_half_life``).
    pretrained_params: Optional pre-trained Haiku parameter pytree to
      *merge* into the freshly initialised params.  Trunk parameter keys
      that exist in ``pretrained_params`` will overwrite the random
      initialisations, effectively loading the pre-trained trunk.
      If ``None``, all parameters (including trunk) are randomly initialised,
      which is only suitable for debugging / smoke tests.
    pretrained_state: Optional Haiku state pytree corresponding to
      ``pretrained_params`` (e.g., BatchNorm running statistics).

  Returns:
    A 4-tuple ``(params, state, opt_state, train_step_fn)`` where:
    - ``params``: Merged model parameters (pre-trained trunk + new head).
    - ``state``: Haiku state (from pre-trained checkpoint if provided).
    - ``opt_state``: Initialised masked optimiser state.
    - ``train_step_fn``: The jitted training step from
      ``get_rna_half_life_train_step``.
  """
  # ---- 1. Initialise all parameters (trunk + head) from scratch ------------
  # We need a dummy RNG key; JAX will trace through the init.
  init_rng = jax.random.PRNGKey(0)
  params, state = forward_fn.init(
      init_rng,
      sample_batch,
      False,  # is_training=False during init
  )

  # ---- 2. Load pre-trained trunk weights (if provided) ---------------------
  if pretrained_params is not None:
    # Merge: update the freshly initialised params with the pre-trained ones.
    # Keys that exist in both use the pre-trained values (trunk).
    # Keys only in params (i.e., the new head) keep their random init.
    #
    # We use jax.tree_util.tree_map_with_path to selectively overwrite trunk
    # keys while preserving the head keys.
    def _merge_leaf(path, fresh_leaf, pretrained_tree):
      """Replaces fresh_leaf with the pre-trained value if available."""
      # Reconstruct the key path string.
      key_parts = []
      for key_entry in path:
        key_parts.append(
            key_entry.key if hasattr(key_entry, 'key') else str(key_entry)
        )
      # Try to navigate pretrained_tree along the same path.
      subtree = pretrained_tree
      try:
        for key_part in key_parts:
          subtree = subtree[key_part]
        # Found a matching leaf in the pre-trained dict → use it.
        return subtree
      except (KeyError, TypeError):
        # Key not found in pre-trained params → keep fresh (random) init.
        return fresh_leaf

    params = jax.tree_util.tree_map_with_path(
        lambda path, leaf: _merge_leaf(path, leaf, pretrained_params),
        params,
    )
    if pretrained_state is not None:
      state = pretrained_state

  # ---- 3. Build the masked training step -----------------------------------
  train_step_fn = get_rna_half_life_train_step(forward_fn, base_optimizer)

  # ---- 4. Initialise the masked optimiser state ----------------------------
  # ``optax.masked`` requires the full parameter pytree (not just the head
  # params) because it needs to allocate state for every parameter slot
  # (even if that state is just a dummy for the frozen trunk parameters).
  mask = _make_trunk_mask(params)
  masked_optimizer = optax.multi_transform(
      {True: base_optimizer, False: optax.set_to_zero()}, mask
  )
  opt_state = masked_optimizer.init(params)

  return params, state, opt_state, train_step_fn
