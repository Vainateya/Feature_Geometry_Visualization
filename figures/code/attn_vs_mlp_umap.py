from src.analysis.visualize import visualizer
from src.analysis.utils import Vis2DDimReductConfig, VisualizeConfig
from src.utils import get_project_root

## NOTE: cuml (gpu accelerated) UMAP is NOT necessarily deterministic. Your results will look slightly different, but should be similar.

save_dir = get_project_root() / "figures" / "attn_vs_mlp_umap"


model_to_data_name = {
    "gpt2": "gpt2_latents-text-pg19-128_samples-1024_sequence_length-identity",
    "llama": "huggyllama-llama-7b_latents-text-pg19-64_samples-2048_sequence_length-identity-DIM_REDUCT_512",
}

for model, data_name in model_to_data_name.items():
    model_save_dir = save_dir / model

    if model == "gpt2":
        layer_filter = lambda meta: meta.pre_add and not meta.is_norm and 1 < meta.block_num < 9
    else:
        layer_filter = lambda meta: meta.pre_add and not meta.is_norm and 5 < meta.block_num < 28

    vis_config = [
        VisualizeConfig(
            data_name=data_name,
            sequence_selection=slice(1, None),
            layer_filter=layer_filter,
            img_save_dir=model_save_dir,
        )
    ]

    vis_dim_reduct_config = [
        Vis2DDimReductConfig(
            # Dim reduct
            dim_reduct_mode="umap",
            gpu_accelerate=True,
            dim_reduct_kwargs={"n_epochs": 100000, "metric": "cosine"},
            n_fit=1e5,
            n_transform=5e5,
            seed=42,
            # Figure
            figsize=(3.25, 3),
            dpi=300,
            subplots_adjust_kwargs={
                "left": 0.02,
                "right": 0.98,
                "top": 0.98,
                "bottom": 0.02,
            },
            disable_labels=True,
            disable_ticks=True,
        )
    ]

    visualizer(vis_config, vis_dim_reduct_config)
