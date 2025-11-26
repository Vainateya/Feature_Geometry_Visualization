from dataclasses import replace

from src.analysis.visualize import visualizer
from src.analysis.utils import Vis2DDimReductConfig, VisualizeConfig
from src.utils import get_project_root

save_dir = get_project_root() / "figures" / "layerwise"

model_to_data_name = {
    "gpt2": "gpt2_latents-text-pg19-128_samples-1024_sequence_length-identity",
    "llama": "huggyllama-llama-7b_latents-text-pg19-64_samples-2048_sequence_length-identity-DIM_REDUCT_512",
}


for model, data_name in model_to_data_name.items():
    model_save_dir = save_dir / model

    vis_config = [
        VisualizeConfig(
            data_name=data_name,
            layer_filter=lambda meta: not meta.pre_add,
            img_save_dir=model_save_dir,
        )
    ]

    vis_dim_reduct_config = [
        Vis2DDimReductConfig(
            convert_unit_vector=True,
            figsize=(3.25, 2.5),
            dpi=300,
            s=0.05,
            subplots_adjust_kwargs={"left": 0.18, "right": 0.98, "top": 0.98, "bottom": 0.18},
        )
    ]

    vis_config.append(replace(vis_config[0], prev_vis_mode=2, img_save_dir=model_save_dir / "mean_samples"))
    vis_dim_reduct_config.append(replace(vis_dim_reduct_config[0], mean_samples=True))

    vis_config.append(replace(vis_config[0], prev_vis_mode=2, img_save_dir=model_save_dir / "mean_seq"))
    vis_dim_reduct_config.append(replace(vis_dim_reduct_config[0], mean_seq=True))

    visualizer(vis_config, vis_dim_reduct_config)
