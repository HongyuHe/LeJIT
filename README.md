# LeJIT: Logic Rule Enforcement for Network Data Generation with LLMs 📐

LeJIT enforces logic formulas during autoregressive LM generation of network key-value records.
It combines:

- [NetNomos](https://github.com/HongyuHe/NetNomos) dataset schemas and rule artifacts
- Hugging Face [causal language models](https://huggingface.co/docs/transformers/tasks/language_modeling)
- Z3-backed stepwise constraint checking during decoding

LeJIT is designed for records such as:

- NetFlow-style flow records
- packet-header windows derived from PCAP traces
- pre-aggregated telemetry and network logs

The Python package and CLI entry point: `lejit`

## Citation

```tex
@inproceedings{he2025lejit,
  title={Just-in-Time Logic Enforcement: A new paradigm of combining statistical and symbolic reasoning for network management},
  author={H{\`e}, Hongyu and Apostolaki, Maria},
  booktitle={Proceedings of the 24th ACM Workshop on Hot Topics in Networks (HotNets 25)},
  pages={184--192},
  year={2025}
}

```

## 1. Introduction

### What LeJIT Does

LeJIT trains a causal language model (LM) over a serialized representation of network records and
filters every generation step against a formal logic theory.

The typical workflow is:

1. Describe the dataset with a NetNomos [dataset schema](https://github.com/HongyuHe/NetNomos#dataset-schema) JSON file.
2. Provide a rule artifact in NetNomos [rule format](https://github.com/HongyuHe/NetNomos#grammar).
3. Train a LeJIT model bundle with `lejit train`.
4. Generate new records with `lejit generate` or complete record prefixes/prompts with `lejit complete`.

### Key Concepts

#### Dataset Schema

LeJIT reuses [NetNomos dataset schemas](https://github.com/HongyuHe/NetNomos#dataset-schema). A dataset schema defines:

- where the input data comes from
- what fields exist and what their value types are
- preprocessing steps such as renaming, mapping, casting, filtering, and hex parsing
- context windows for packet-local or sequence-local reasoning
- derived variables such as interarrival statistics

#### Rule Artifact

LeJIT consumes NetNomos rule artifacts from `rules.json`. Each rule is a structured logic formula,
not a free-form string. The rule bundle can be learned with NetNomos or supplied manually in the
same JSON format.

#### Model Bundle

`lejit train` writes a self-contained bundle directory with:

- `model/`: Hugging Face model weights and config
- `config.json`: resolved LeJIT config
- `dataset_spec.json`: embedded dataset schema
- `rules.json`: embedded rule artifact
- `metadata.json`: rule metadata when present
- `schema.json`: LeJIT field serialization schema
- `vocab.json`: LeJIT vocabulary
- `manifest.json`: summary metadata

Generation and completion load their model, schema, and embedded rules from the bundle.

## 2. Installation & Setup

### Requirements

- Python `>=3.10`
- [`uv`](https://docs.astral.sh/uv/) for environment management and command execution
- A local checkout of NetNomos because this project depends on the `netnomos` Python package via
  a local `uv` source

### Recommended Layout

The current [`pyproject.toml`](pyproject.toml) expects `NetNomos` to live beside this repository
for dependency resolution:

```text
<workspace>/
├── LeJIT/
└── NetNomos/
```

### Setup

```bash
git clone https://github.com/HongyuHe/LeJIT
git clone https://github.com/HongyuHe/NetNomos

cd LeJIT
uv sync
```

If your `NetNomos` checkout is not at `../NetNomos`, update `[tool.uv.sources]` in
[`pyproject.toml`](pyproject.toml).

### Verify the Environment

```bash
uv run lejit --help
uv run pytest -q
```

### Repository Locations

- local datasets: [`data/`](data)
- local dataset schemas: [`data/specs/`](data/specs)
- local rules: [`rules/`](rules)
- preset LeJIT configs: [`configs/`](configs)
- thin dataset wrappers: [`scripts/`](scripts)
- library code: [`lejit/`](lejit)

The shipped configs currently use local paths for all dataset assets:

- `dataset_spec` points into [`data/specs/`](data/specs)
- `input_path` points into [`data/`](data)
- `rules_path` points into [`rules/`](rules)

NetNomos is still needed at install time because the Python package is imported directly by LeJIT.

## 3. Supported Language Models

LeJIT builds models with `transformers.AutoModelForCausalLM` of [Hugging Face](https://huggingface.co/docs/transformers/en/index) in
[`lejit/modeling.py`](lejit/modeling.py). In practice, this means:

- supported: causal language model architectures available through
  `AutoModelForCausalLM.from_config(...)` or `AutoModelForCausalLM.from_pretrained(...)`
- intended scope: decoder-style next-token models
- not supported: encoder-only masked language models, translation-style seq2seq models, or models
  that are not exposed through `AutoModelForCausalLM`

Examples of compatible decoder-only checkpoints include:

- `gpt2`
- `distilgpt2`
- `EleutherAI/gpt-neo-125M`
- `EleutherAI/gpt-j-6B`
- `facebook/opt-125m`
- etc.

For `model.mode = "config"`, common `architecture` values include:

- `"gpt2"`
- `"gpt_neo"`
- `"gptj"`
- `"opt"`
- etc.

Other decoder-only families such as LLaMA-style or Mistral-style models may also work when they
are available through the Hugging Face `AutoModelForCausalLM` registry, but the most direct
starting point in this repository is still a GPT-2-family or GPT-Neo/OPT-family configuration.

Important implementation detail:

- LeJIT does **not** use a Hugging Face tokenizer
- it builds its own vocabulary from serialized field/value tokens
- the model embedding matrix is resized to that LeJIT vocabulary

Because of that, `model.mode = "config"` is the most natural starting point for training.
`model.mode = "pretrained"` is supported when you intentionally want to initialize from an
existing causal LM architecture before resizing embeddings to the LeJIT vocabulary.

## 4. Configuration Files

Each CLI command takes a TOML config parsed by
[`LeJITConfig`](lejit/config.py). The top-level sections are:

<details>
<summary><code>[dataset]</code> options</summary>

- `dataset_spec`: path to a NetNomos dataset schema JSON file
- `input_path`: path to the raw dataset file; overrides `source.path` from the dataset schema
- `rules_path`: path to the NetNomos `rules.json` artifact
- `limit`: optional row or packet limit for smoke tests

</details>

<details>
<summary><code>[model]</code> options</summary>

- `mode`: `"config"` or `"pretrained"`
- `architecture`: model type key used when building from config, such as `"gpt2"`
- `name_or_path`: required in `"pretrained"` mode; optional in `"config"` mode when loading a base
  config from an existing checkpoint
- `config_overrides`: model config overrides such as `n_positions`, `n_embd`, `n_layer`, and
  `n_head`

</details>

<details>
<summary><code>[serialization]</code> options</summary>

- `field_order`: optional explicit field order for serialization and prompting
- `max_categorical_domain`: optional hard stop for very large categorical domains
- `force_string_fields`: reserved for future schema coercion
- `numeric_precision`: decimal precision used when rendering real-valued fields

</details>

<details>
<summary><code>[training]</code> options</summary>

- `epochs`
- `batch_size`
- `learning_rate`
- `weight_decay`
- `warmup_ratio`
- `gradient_accumulation_steps`
- `max_steps`
- `seed`
- `logging_steps`
- `save_steps`

</details>

<details>
<summary><code>[decoding]</code> options</summary>

- `max_new_tokens`
- `temperature`
- `top_k`
- `top_p`
- `do_sample`
- `backtrack_limit`
- `num_return_sequences`

</details>

<details>
<summary><code>[run]</code> options</summary>

- `n_samples`: default sample count for `generate`
- `batch_size`: currently reserved for higher-level orchestration
- `samples_per_prompt`: default fan-out for `complete`
- `prompt_columns`: reserved for future prompt presets

</details>

## 5. CLI Usage

### `lejit --help`

<details>
<summary>Show command summary</summary>

```text
usage: lejit [-h] {train,generate,complete} ...

Train and run LeJIT.

positional arguments:
  {train,generate,complete}
    train               Train a LeJIT bundle.
    generate            Generate constrained rows.
    complete            Complete prefix prompts.
```

</details>

### `lejit train`

Purpose:

- load the dataset through NetNomos preprocessing
- derive the LeJIT serialization schema and vocabulary
- build and train a causal LM
- save a self-contained bundle

<details>
<summary>Options</summary>

- `--config`: required TOML config
- `--output`: required bundle directory

</details>

<details>
<summary>Example</summary>

```bash
uv run lejit train \
  --config configs/cidds/train.toml \
  --output artifacts/cidds-model
```

</details>

### `lejit generate`

Purpose:

- load a trained bundle
- instantiate the embedded schema, vocabulary, and rule theory
- generate full records under rule enforcement

<details>
<summary>Options</summary>

- `--config`: required TOML config
- `--model-bundle`: required trained bundle
- `--output`: required CSV output path
- `--n-samples`: optional override for `[run].n_samples`
- `--device`: device string such as `cpu`, `cuda`, or `mps`

</details>

Behavior note:

- the bundle supplies the trained model, schema, and embedded rule artifact
- the external config is mainly useful for decoding and run defaults at generation time

<details>
<summary>Example</summary>

```bash
uv run lejit generate \
  --config configs/netflix/generate.toml \
  --model-bundle artifacts/netflix-model \
  --output out/netflix.csv \
  --n-samples 1000 \
  --device cpu
```

</details>

### `lejit complete`

Purpose:

- load a trained bundle
- accept prompt records from CSV
- complete the remaining fields under rule enforcement

<details>
<summary>Options</summary>

- `--config`: required TOML config
- `--model-bundle`: required trained bundle
- `--prompts`: required CSV file
- `--output`: required CSV output path
- `--device`: device string such as `cpu`, `cuda`, or `mps`
- `--samples-per-prompt`: optional override for `[run].samples_per_prompt`

</details>

Important prompt contract:

<details>
<summary>Prompt prefix rules</summary>

- prompt columns must be a **strict left prefix** of the active serialization order
- if the schema order is `[A, B, C, D]`, valid prompts may contain `[A]`, `[A, B]`, or
  `[A, B, C]`
- prompts like `[B, C]` are rejected

</details>

<details>
<summary>Example</summary>

```bash
uv run lejit complete \
  --config configs/metadc/complete.toml \
  --model-bundle artifacts/metadc-model \
  --prompts prompts/metadc_prefixes.csv \
  --output out/metadc_completed.csv \
  --samples-per-prompt 4 \
  --device cpu
```

</details>

### Dataset-Specific Wrapper Scripts

The repository also ships thin wrappers such as:

- [`scripts/generate_cidds.py`](scripts/generate_cidds.py)
- [`scripts/complete_cidds.py`](scripts/complete_cidds.py)
- [`scripts/generate_netflix.py`](scripts/generate_netflix.py)
- [`scripts/complete_netflix.py`](scripts/complete_netflix.py)

These wrappers only preselect a config and forward to the main CLI.

## 6. Python API

The main entry points are exported from [`lejit/__init__.py`](lejit/__init__.py):

- `LeJITConfig`
- `LeJITPipeline`

### Load a Config

```python
from lejit import LeJITConfig

config = LeJITConfig.from_toml("configs/cidds/train.toml")
```

`LeJITConfig.from_toml(...)` validates the full TOML file with Pydantic and returns a structured
object you can inspect or modify in code.

### Build and Train a Pipeline

```python
from lejit import LeJITConfig, LeJITPipeline

config = LeJITConfig.from_toml("configs/cidds/train.toml")
pipeline = LeJITPipeline.build_from_config(
    config,
    base_dir="configs/cidds",
)
bundle_dir = pipeline.train("artifacts/cidds-model")
```

`LeJITPipeline.build_from_config(...)` does the following:

- resolves config-relative paths
- loads the dataset via NetNomos
- loads and validates the rule artifact
- derives the LeJIT schema and vocabulary
- builds a Hugging Face causal LM

`pipeline.train(...)` then:

- serializes each prepared record into a LeJIT token sequence
- trains the model with `transformers.Trainer`
- saves a bundle directory

### Load a Trained Bundle

```python
from lejit import LeJITPipeline

pipeline = LeJITPipeline.load("artifacts/cidds-model", device="cpu")
```

`LeJITPipeline.load(...)` restores:

- the trained model
- the saved vocabulary
- the saved LeJIT schema
- the embedded dataset schema and rule artifact

### Generate Full Records

```python
frame = pipeline.generate(
    n_samples=100,
    device="cpu",
)
```

Return type:

- `pandas.DataFrame`

### Complete Prefix Records

```python
import pandas as pd

prompts = pd.read_csv("prompts.csv")
completed = pipeline.complete(
    prompts,
    samples_per_prompt=4,
    device="cpu",
)
```

Return type:

- `pandas.DataFrame`

`complete(...)` enforces the same prefix rule as the CLI: prompt columns must match the leftmost
portion of the active field order.

## 7. Adding a New Dataset

### Step 1: Place the Data

Put the raw data file under [`data/`](data) or another location reachable by your config.

Examples:

- `data/my_flows.csv`
- `data/my_trace.pcap`
- `data/my_logs.csv`

### Step 2: Write a Dataset Schema

Create a NetNomos dataset schema JSON file. You may:

- keep it under [`data/specs/`](data/specs)
- or store it elsewhere and point `dataset_spec` at that path

A minimal CSV example looks like:

```json
{
  "name": "my_dataset",
  "source": {
    "type": "csv",
    "path": "data/my_flows.csv"
  },
  "fields": [
    {"name": "SrcIp", "value_type": "categorical", "roles": ["src", "identifier"]},
    {"name": "DstIp", "value_type": "categorical", "roles": ["dst", "identifier"]},
    {"name": "Bytes", "value_type": "integer", "roles": ["size", "measurement"]}
  ]
}
```

Add the richer NetNomos fields you need:

- `preprocessing`
- `context_window`
- `derived_variables`
- explicit `domain` or `bounds`
- semantic `roles`

### Step 3: Provide a Rule Artifact

Create a rule directory such as `rules/my_dataset/` containing:

- `rules.json`
- `metadata.json`
- optionally `interpreted_rules.clj`

`rules.json` must follow the NetNomos learned-rule format. Each entry is a structured formula with
fields like:

- `rule_id`
- `formula`
- `display`
- `support`
- `source`

LeJIT validates rule references against the prepared dataset fields before training starts.

### Step 4: Add LeJIT TOML Configs

Create:

- `configs/my_dataset/train.toml`
- `configs/my_dataset/generate.toml`
- `configs/my_dataset/complete.toml`

Minimal training config:

<details>
<summary>Show example <code>train.toml</code></summary>

```toml
[dataset]
dataset_spec = "../../data/specs/my_dataset.json"
input_path = "../../data/my_flows.csv"
rules_path = "../../rules/my_dataset/rules.json"

[model]
mode = "config"
architecture = "gpt2"

[model.config_overrides]
n_positions = 512
n_ctx = 512
n_embd = 256
n_layer = 6
n_head = 8

[serialization]
numeric_precision = 6
max_categorical_domain = 50000

[training]
epochs = 3
batch_size = 16
learning_rate = 0.0005
seed = 42

[decoding]
temperature = 1.0
do_sample = true
top_p = 0.95

[run]
n_samples = 100
samples_per_prompt = 1
```

</details>

### Step 5: Tune Serialization Choices

Pay attention to:

- `field_order`
  This controls serialization order and therefore prompt legality in `complete`.
- `max_categorical_domain`
  Use this to catch exploding categorical vocabularies early.
- `numeric_precision`
  This affects how real-valued fields are rendered into fixed-width token streams.

If a categorical field is too large, consider:

- preprocessing it into a coarser representation
- mapping values into semantic classes
- using a bounded integer encoding in the dataset schema

### Step 6: Train and Smoke-Test

```bash
uv run lejit train \
  --config configs/my_dataset/train.toml \
  --output artifacts/my_dataset-model

uv run lejit generate \
  --config configs/my_dataset/generate.toml \
  --model-bundle artifacts/my_dataset-model \
  --output out/my_dataset.csv \
  --n-samples 10
```

### Step 7: Prepare Completion Prompts

If you want completion:

1. inspect the active field order
2. create a prompt CSV containing only the leftmost `k` fields
3. keep the exact column order

Example:

- if the field order is `[rackid, hostid, IngressBytesAgg, EgressBytesAgg]`
- valid prompt CSV columns are:
  - `[rackid]`
  - `[rackid, hostid]`
  - `[rackid, hostid, IngressBytesAgg]`

## 8. Current Presets

The repository currently ships preset configs for:

- [CIDDS](https://www.hs-coburg.de/forschen/cidds-coburg-intrusion-detection-data-sets/) NetFlow records
- [Netflix](https://dl.acm.org/doi/10.1145/3366704) PCAP trace
- [MAWI](https://mawi.wide.ad.jp/mawi/) PCAP trace
- preprocessed datacenter logs from Meta [[IMC '22](https://dl.acm.org/doi/10.1145/3517745.3561430)]

See [`configs/`](configs) and [`scripts/`](scripts).
