# Exploring LLM Latent Space Geometry Through Dimensionality Reduction


## Environment

Install:
```bash
conda env create -f environment.yml && conda activate vis-llm-latent
```


## Generating and Saving Latent State Data

Note: Ensure your system has sufficient storage and memory. The `text` mode latent states can take a few hundred GB of both storage and RAM. In the worst case (LLaMa latent states w/o dim reduction), at least ~260 GB of storage and ~600 GB of RAM are required to generate, save, and visualize.

### Text Mode PG-19 - GPT-2

```python
python src/generation/main.py \
    --model_name="gpt2" \
    --dataset="pg19" \
    --sequence_length=1024 \
    --num_inputs=128
```

### Text Mode PG-19 - LLaMa

Generate and save latent states:

```python
python src/generation/main.py \
    --model_name="huggyllama/llama-7b" \
    --dataset="pg19" \
    --sequence_length=2048 \
    --num_inputs=64 \
    --skip_norm_capture=True
```

Reduce dim from 4096 to 512:

```python
python src/generation/dim_reduct.py \
    --data_name="huggyllama-llama-7b_latents-text-64_samples-2048_sequence_length-identity" \
    --new_dim=512 \
    --n_fit=4000000 \
    --n_fit_samples=32
```

You can reduce `n_fit` and `n_fit_samples` if you are experiencing out-of-memory issues. Reducing `n_fit_samples` will likely be most helpful.

### Singular Mode - GPT-2

```python
python src/generation/main.py \
    --model_name="gpt2" \
    --mode="singular" \
    --num_inputs=None
```

### Singular Mode - LLaMa

```python
python src/generation/main.py \
    --model_name="huggyllama/llama-7b" \
    --mode="singular" \
    --num_inputs=None
```

## Reproducing Dim Reduct Visualizations

To reproduce the visualizations, please run the `.py` files under `figures/code`.

## Reproducing Norm Plots

To reproduce norm plots, please use the notebooks in `notebooks`.