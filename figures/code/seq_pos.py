import matplotlib.pyplot as plt


from src.analysis.visualize import visualizer
from src.analysis.utils import Vis2DDimReductConfig, VisualizeConfig
from src.utils import get_project_root

save_dir = get_project_root() / "figures" / "seq_pos"


model_to_data_name = {
    "gpt2": "gpt2_latents-text-pg19-128_samples-1024_sequence_length-identity",
    "llama": "huggyllama-llama-7b_latents-text-pg19-64_samples-2048_sequence_length-identity-DIM_REDUCT_512",
}

for model, data_name in model_to_data_name.items():
    model_save_dir = save_dir / model

    if model == "gpt2":
        layer_filter = lambda meta: not meta.pre_add and 1 < meta.block_num < 9
    else:
        layer_filter = lambda meta: not meta.pre_add and 5 < meta.block_num < 28

    for convert_unit_vector in (True, False):
        unit_save_dir = model_save_dir / f"unit_{convert_unit_vector}"

        vis_config = [
            VisualizeConfig(
                data_name=data_name,
                sequence_selection=slice(1, None) if not convert_unit_vector else None,
                layer_filter=layer_filter,
                img_save_dir=unit_save_dir,
                do_many_vis=True,
                composite_vis_grid_shape=(5, 3),
            )
        ]

        vis_dim_reduct_config = [
            Vis2DDimReductConfig(
                ## Dim reduct
                mean_layers=True,
                mean_samples=True,
                n_components=6,
                convert_unit_vector=convert_unit_vector,
                ## Figure
                figsize=(6.5 / 3, 1.25),
                dpi=300,
                fontsize=6,
                labelsize=5,
                subplots_adjust_kwargs={"left": 0.20, "right": 0.98, "top": 0.95, "bottom": 0.23},
                yaxis_formatter=plt.FuncFormatter(lambda x, p: f"{x:>4.1f}".replace("-", "\u2212")),
            )
        ]

        visualizer(vis_config, vis_dim_reduct_config)
