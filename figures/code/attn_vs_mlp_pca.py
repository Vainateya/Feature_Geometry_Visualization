from dataclasses import replace


import matplotlib.pyplot as plt


from src.analysis.visualize import visualizer
from src.analysis.utils import Vis2DDimReductConfig, VisualizeConfig
from src.utils import get_project_root

save_dir = get_project_root() / "figures" / "attn_vs_mlp_pca"


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

    for convert_unit_vector in (True, False):
        unit_save_dir = model_save_dir / f"unit_{convert_unit_vector}"

        vis_config = [
            VisualizeConfig(
                data_name=data_name,
                sequence_selection=slice(1, None),
                layer_filter=layer_filter,
                img_save_dir=unit_save_dir,
            )
        ]

        vis_dim_reduct_config = [
            Vis2DDimReductConfig(
                convert_unit_vector=convert_unit_vector,
                figsize=(3, 2),
                dpi=300,
                subplots_adjust_kwargs={"left": 0.15, "right": 0.95, "top": 0.98, "bottom": 0.12},
                disable_labels=True,
                yaxis_formatter=plt.FuncFormatter(lambda x, p: f"{x:.1f}".replace("-", "\u2212")),
            )
        ]

        vis_config.append(replace(vis_config[0], prev_vis_mode=2, img_save_dir=unit_save_dir / "mean_samples"))
        vis_dim_reduct_config.append(replace(vis_dim_reduct_config[0], mean_samples=True))

        visualizer(vis_config, vis_dim_reduct_config)
