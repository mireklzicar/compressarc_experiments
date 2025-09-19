import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

from tqdm import tqdm

import numpy as np
import torch
import matplotlib.pyplot as plt

import train
import preprocessing
import arc_compressor
import initializers
import multitensor_systems
import layers
import solution_selection
import visualization


"""
This file allows you to train one model on one task, and see plots of what
the process and end result looks like. You can input the training split and
the task code, and it will:
- Train a model for 1500 steps,
- Plot sampled solutions from the model at every 50 steps,
- Plot the KL and reconstruction error over time,
- Plot the contribution of each tensor shape to the KL over time,
- Show top principal components of each tensor that still contributes to
  the KL at the end of training.
"""

# For some reason trying to set the seed doesn't actually fix results.
# Just run things over and over again until you see desired interesting behaviors.
np.random.seed(0)
torch.manual_seed(0)
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')


SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_SUBPATH = Path("checkpoints/latent_test_holdout_multitask/dcgru/dngpu_best.pt")


def _resolve_default_checkpoint() -> Optional[Path]:
    """Search common locations for the pretrained CellGRU checkpoint."""
    roots = [SCRIPT_DIR]
    parents = list(SCRIPT_DIR.parents)
    roots.extend(parents[:4])
    for parent in parents[:4]:
        roots.append(parent / "arc-cellgru")

    seen = set()
    for root in roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        candidate = root / CHECKPOINT_SUBPATH
        if candidate.exists():
            return candidate
    return None


DEFAULT_CHECKPOINT = _resolve_default_checkpoint()


def _clean_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove torch.compile artefact prefixes from checkpoint keys."""
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key.replace("._orig_mod.", ".")
        cleaned[new_key] = value
    return cleaned


def _load_cellgru_state(path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    ckpt = torch.load(path, map_location="cpu")
    state = _clean_state_dict(ckpt["model_state"])
    args = ckpt.get("args", {})
    return state, args


def _remap_checkpoint_vocab(state: Dict[str, torch.Tensor], task, target_vocab: int) -> Dict[str, torch.Tensor]:
    """Align checkpoint vocab channels to the task-local token order."""
    state = dict(state)  # shallow copy

    def _build_mapping() -> Tuple[int, ...]:
        # Token 0 reserved for padding; subsequent tokens follow task.colors order.
        mapping = [0]
        for color in task.colors:
            mapping.append(int(color) + 1)
        return tuple(mapping[:target_vocab])

    if "input_layer.weight" not in state or "output_layer.weight" not in state:
        return state

    mapping = _build_mapping()
    ckpt_vocab = state["input_layer.weight"].shape[1]
    if ckpt_vocab == target_vocab:
        return state

    weight = state["input_layer.weight"]
    new_weight = weight.new_zeros(weight.shape[0], target_vocab, 1, 1)
    for idx, src_idx in enumerate(mapping):
        if src_idx < ckpt_vocab:
            new_weight[:, idx, :, :] = weight[:, src_idx, :, :]
    state["input_layer.weight"] = new_weight

    if "output_layer.weight" in state:
        out_w = state["output_layer.weight"]
        new_out_w = out_w.new_zeros(target_vocab, out_w.shape[1], 1, 1)
        for idx, src_idx in enumerate(mapping):
            if src_idx < out_w.shape[0]:
                new_out_w[idx, :, :, :] = out_w[src_idx, :, :, :]
        state["output_layer.weight"] = new_out_w

    if "output_layer.bias" in state:
        out_b = state["output_layer.bias"]
        new_out_b = out_b.new_zeros(target_vocab)
        for idx, src_idx in enumerate(mapping):
            if src_idx < out_b.shape[0]:
                new_out_b[idx] = out_b[src_idx]
        state["output_layer.bias"] = new_out_b

    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ARCCompressor on a single ARC task and visualize progress.")
    parser.add_argument("--split", type=str, default=None,
                        help="Dataset split to use (training/evaluation/test). Defaults to prompting.")
    parser.add_argument("--task", type=str, default=None,
                        help="Task ID hash (e.g. 272f95fa). Defaults to prompting.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained CellGRU checkpoint. Use 'default' to load the bundled checkpoint."
                             " If omitted you will be prompted.")
    parser.add_argument("--steps", type=int, default=1500,
                        help="Number of training iterations to run (default: 1500).")
    parser.add_argument("--plot-interval", type=int, default=50,
                        help="Draw solution snapshots every N steps (default: 50).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to store outputs. Defaults to <task>/ under the current directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Some interesting tasks: 272f95fa, 6d75e8bb, 6cdd2623, 41e4d17e, 2bee17df
    # 228f6490, 508bd3b6, 2281f1f4, ecdecbb3
    split = args.split or input('Enter which split you want to find the task in (training, evaluation, test): ')
    task_name = args.task or input('Enter which task you want to analyze (eg. 272f95fa): ')

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = Path(task_name)
    folder = str(out_dir) + '/'
    print('Performing a training run on task', task_name,
          'and placing the results in', folder)
    os.makedirs(folder, exist_ok=True)

    # Preprocess the task, set up the training
    task = preprocessing.preprocess_tasks(split, [task_name])[0]
    model = arc_compressor.ARCCompressor(task)

    if args.checkpoint is not None:
        ckpt_input = args.checkpoint.strip()
    else:
        ckpt_input = input(
            "Enter CellGRU checkpoint path to preload (blank to skip, 'default' for pretrained): "
        ).strip()
    checkpoint_path: Optional[Path] = None
    if ckpt_input:
        if ckpt_input.lower() in {"default", "d"}:
            if DEFAULT_CHECKPOINT is not None:
                checkpoint_path = DEFAULT_CHECKPOINT
                print(f"Using default checkpoint at {checkpoint_path}")
            else:
                print("Default checkpoint not found; continuing without preload.")
        else:
            checkpoint_path = Path(ckpt_input).expanduser().resolve()
            if not checkpoint_path.exists():
                print(f"Checkpoint {checkpoint_path} not found; continuing without preload.")
                checkpoint_path = None

    if checkpoint_path is not None:
        try:
            state_dict, train_args = _load_cellgru_state(checkpoint_path)
            state_dict = _remap_checkpoint_vocab(state_dict, task, model.n_vocab)
            net_state = model.net.state_dict()
            filtered_state = {k: v for k, v in state_dict.items() if k in net_state}
            missing_keys = [k for k in net_state if k not in state_dict]
            unexpected_keys = [k for k in state_dict if k not in net_state]
            load_result = model.net.load_state_dict(filtered_state, strict=False)
            if load_result.missing_keys:
                missing_keys.extend(load_result.missing_keys)
            if load_result.unexpected_keys:
                unexpected_keys.extend(load_result.unexpected_keys)
            print(f"Loaded pretrained CellGRU weights from {checkpoint_path}.")
            param_millions = sum(p.numel() for p in model.net.parameters()) / 1e6
            print(f"Model size: {param_millions:.2f}M parameters")
            if missing_keys:
                print("Missing keys (ignored):", missing_keys)
            if unexpected_keys:
                print("Unexpected keys (ignored):", unexpected_keys)

            default_steps = int(train_args.get("steps", train_args.get("dngpu_steps", model.steps)))
            model.steps = default_steps
            print(f"Default recurrent steps from checkpoint: {default_steps}")
        except Exception as exc:
            print(f"Failed to load checkpoint {checkpoint_path}: {exc}")

    optimizer = torch.optim.Adam(model.weights_list, lr=3e-4)  # betas default are fine
    train_history_logger = solution_selection.Logger(task)
    visualization.plot_problem(train_history_logger)

    # Perform training for the requested number of iterations
    plot_interval = max(1, int(args.plot_interval))
    n_iterations = max(1, int(args.steps))
    for train_step in tqdm(range(n_iterations)):
        train.take_step(task, model, optimizer, train_step, train_history_logger)
        
        # Plot solutions periodically
        if (train_step + 1) % plot_interval == 0:
            visualization.plot_solution(train_history_logger,
                fname=folder + task_name + '_at_' + str(train_step+1) + ' steps.png')
            visualization.plot_solution(train_history_logger,
                fname=folder + task_name + '_at_' + str(train_step+1) + ' steps.pdf')

    # Save the metrics, model weights, and learned representations.
    np.savez(folder + task_name + '_KL_curves.npz',
             KL_curves={key:np.array(val) for key, val in train_history_logger.KL_curves.items()},
             reconstruction_error_curve=np.array(train_history_logger.reconstruction_error_curve),
             multiposteriors=model.multiposteriors,
             target_capacities=model.target_capacities,
             decode_weights=model.decode_weights)

    # Load the metrics, model weights, and learned representations.
    stored_data = np.load(folder + task_name + '_KL_curves.npz', allow_pickle=True)
    KL_curves = stored_data['KL_curves'][()]
    reconstruction_error_curve = stored_data['reconstruction_error_curve']
    multiposteriors = stored_data['multiposteriors'][()]
    target_capacities = stored_data['target_capacities'][()]
    decode_weights = stored_data['decode_weights'][()]
    
    # Plot the KL curves over time.
    # For specific tasks that we found interesting, we wrote some
    # code to color interesting KL components differently.
    special_curve_colors = {
            '272f95fa': {
                'dims': [(1,0,0,1,0), (1,0,0,0,1), (0,1,1,0,0), (0,1,0,0,0)],
                'colors': [(1, 0, 0), (0, 1, 0), (0, 0.5, 1), (0.5, 0, 1)]
            },
            '6cdd2623': {
                'dims': [(1,0,0,1,0), (1,0,0,0,1), (1,1,0,0,0), (1,0,0,1,1), (0,0,1,0,0)],
                'colors': [(1, 0.6, 0), (0, 1, 0), (0, 0.5, 1), (0.5, 0, 1), (1, 0, 0.5)]
            },
            '41e4d17e': {
                'dims': [(1,0,0,1,1), (0,1,0,0,0)],
                'colors': [(1, 0, 0), (0, 0, 1)]
            },
            '6d75e8bb': {
                'dims': [(1,0,0,1,0), (1,0,0,0,1), (1,0,0,1,1), (0,1,0,0,0)],
                'colors': [(1, 0, 0), (0, 1, 0), (0, 0.5, 1), (0.5, 0, 1)]
            }
    }
    fig, ax = plt.subplots()
    for component_name, curve in KL_curves.items():
        line_color = (0.5, 0.5, 0.5)
        label = None
        if task_name in special_curve_colors:
            dims_list = special_curve_colors[task_name]['dims']
            colors_list = special_curve_colors[task_name]['colors']
            for dims, color in zip(dims_list, colors_list):
                if tuple(eval(component_name)) == dims:
                    line_color = color
                    axis_names = ['example', 'color', 'direction', 'height', 'width']
                    axis_names = [axis_name
                        for axis_name, axis_exists in zip(axis_names, dims) if axis_exists]
                    label = '(' + ', '.join(axis_names) + ', channel)'
        ax.plot(np.arange(curve.shape[0]), curve, color=line_color, label=label)
    if task_name == '6cdd2623':
        ax.set_ylim((0.3, 4e4))

    ax.legend()
    plt.yscale('log')
    plt.xlabel('step')
    plt.ylabel('KL contribution')
    ax.grid(which='both', linestyle='-', linewidth='0.5', color='gray')
    plt.savefig(folder + task_name + '_KL_components.png', bbox_inches='tight')
    plt.close()

    # Plot the KL vs reconstruction error
    fig, ax = plt.subplots()
    total_KL = 0
    for component_name, curve in KL_curves.items():
        total_KL = total_KL + curve
    fig, ax = plt.subplots()
    ax.plot(np.arange(total_KL.shape[0]), total_KL, label='KL from z', color='k')
    ax.plot(np.arange(reconstruction_error_curve.shape[0]),
            reconstruction_error_curve, label='reconstruction error', color='r')
    ax.legend()
    plt.yscale('log')
    plt.xlabel('step')
    plt.ylabel('total KL or reconstruction error')
    ax.grid(which='both', linestyle='-', linewidth='0.5', color='gray')
    plt.savefig(folder + task_name + '_KL_vs_reconstruction.png', bbox_inches='tight')
    plt.close()

    # Get the learned representation tensors
    samples = []
    for i in range(100):
        sample, KL_amounts, KL_names = layers.decode_latents(target_capacities,
                                           decode_weights, multiposteriors)
        samples.append(sample)

    def average_samples(dims, *items):
        mean = torch.mean(torch.stack(items, dim=0), dim=0).detach().cpu().numpy()
        all_but_last_dim = tuple(range(len(mean.shape) - 1))
        mean = mean - np.mean(mean, axis=all_but_last_dim)
        return mean
    means = multitensor_systems.multify(average_samples)(*samples)

    # Figure out which tensors contain significant information
    dims_to_plot = []
    for KL_amount, KL_name in zip(KL_amounts, KL_names):
        dims = tuple(eval(KL_name))
        if torch.sum(KL_amount).detach().cpu().numpy() > 1:
            dims_to_plot.append(dims)

    # Show the top principal components of the significant tensors.
    color_names = ['black', 'blue', 'red', 'green', 'yellow', 'gray', 'magenta', 'orange', 'light blue', 'brown']
    restricted_color_names = [color_names[i] for i in task.colors]
    restricted_color_codes = [tuple((visualization.color_list[i]/255).tolist())
                              for i in task.colors]
    for dims in dims_to_plot:
        tensor = means[dims]

        orig_shape = tensor.shape
        if len(orig_shape) == 2:
            tensor = tensor[None,:,:]
        orig_shape = tensor.shape
        if len(orig_shape) == 3:
            tensor = np.reshape(tensor, (-1, orig_shape[-1]))
            U, S, Vh = np.linalg.svd(tensor)  # Get top 3 principal components
            for component_num in range(3):
                component = np.reshape(U[:,component_num], orig_shape[:-1])
                component = component / np.max(np.abs(component))
                strength = S[component_num] / tensor.shape[0]  # Calculate component strength

                # Show the component
                fig, ax = plt.subplots()
                ax.imshow(component, cmap='gray', vmin=-1, vmax=1)
                
                # Pick the axis labels
                axis_names = ['example', 'color', 'direction', 'height', 'width']
                tensor_name = '_'.join([axis_name
                    for axis_name, axis_exists in zip(axis_names, dims) if axis_exists])
                if sum(dims) == 2:
                    x_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][0]
                    y_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][1]
                else:
                    x_dim = None
                    y_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][0]
                plt.ylabel(x_dim)
                plt.xlabel(y_dim)

                if x_dim is None:
                    ax.set_yticks([])
                    ax.set_xticks([], minor=True)
                if y_dim is None:
                    ax.set_xticks([])
                    ax.set_xticks([], minor=True)

                # Set the tick labels
                # Tick labels for example axis
                if x_dim == 'example':
                    ax.set_yticks(np.arange(task.n_examples))
                if y_dim == 'example':
                    ax.set_xticks(np.arange(task.n_examples))

                # Tick labels for color axis
                if x_dim == 'color':
                    ax.set_yticks(np.arange(len(restricted_color_names[1:])))
                    ax.set_yticklabels(restricted_color_names[1:])
                    for ticklabel, tickcolor in zip(ax.get_yticklabels(), restricted_color_codes[1:]):
                        ticklabel.set_color(tickcolor)
                        ticklabel.set_fontweight("bold")
                if y_dim == 'color':
                    ax.set_xticks(np.arange(len(restricted_color_names[1:])))
                    ax.set_xticklabels(restricted_color_names[1:])
                    for ticklabel, tickcolor in zip(ax.get_xticklabels(), restricted_color_codes[1:]):
                        ticklabel.set_color(tickcolor)
                        ticklabel.set_fontweight("bold")

                # Tick labels for direction axis
                direction_names = ["↓", "↘", "→", "↗", "↑", "↖", "←", "↙"]
                if x_dim == 'direction':
                    ax.set_yticks(np.arange(8))
                    ax.set_yticklabels(direction_names)
                    ax.tick_params(axis='y', which='major', labelsize=22)
                if y_dim == 'direction':
                    ax.set_xticks(np.arange(8))
                    ax.set_xticklabels(direction_names)
                    ax.tick_params(axis='x', which='major', labelsize=22)

                # Standard tick labels for height and width axes

                ax.set_title('component' + str(component_num) + ', strength = ' + str(float(strength)))
                plt.savefig(folder + task_name + '_' + tensor_name + '_component_' + str(component_num) + '.png', bbox_inches='tight')
                plt.close()

        # Plot an ({example, color, direction}, x, y) tensor with subplots
        elif len(orig_shape) == 4 and dims[3] == 1 and dims[4] == 1:
            tensor = np.reshape(tensor, (-1, orig_shape[-1]))
            U, S, Vh = np.linalg.svd(tensor)  # Get the top 3 principal components
            for component_num in range(3):
                component = np.reshape(U[:,component_num], orig_shape[:-1])
                component = component / np.max(np.abs(component))
                strength = S[component_num] / tensor.shape[0]
                n_plots = orig_shape[0]

                # Make the subplots
                fig, axs = plt.subplots(1, n_plots)
                for plot_idx in range(n_plots):
                    ax = axs[plot_idx]
                    ax.imshow(component[plot_idx,:,:], cmap='gray', vmin=-1, vmax=1)

                    # Get the axis labels
                    axis_names = ['example', 'color', 'direction', 'height', 'width']
                    tensor_name = '_'.join([axis_name
                        for axis_name, axis_exists in zip(axis_names, dims) if axis_exists])
                    ax_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][0]
                    x_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][1]
                    y_dim = [axis_names[i] for i, dim in enumerate(dims) if dim][2]
                    ax.set_ylabel(x_dim)
                    ax.set_xlabel(y_dim)

                    # Standard tick labels for height and width axes

                    # Label the subplots
                    if ax_dim == 'example':
                        ax.set_title('example ' + str(plot_idx))
                    elif ax_dim == 'color':
                        ax.set_title(restricted_color_names[plot_idx],
                                     color=restricted_color_codes[plot_idx],
                                     fontweight="bold")
                    elif ax_dim == 'direction':
                        direction_names = ["↓", "↘", "→", "↗", "↑", "↖", "←", "↙"]
                        ax.set_title(direction_names[plot_idx], fontsize=22)

                plt.subplots_adjust(wspace=1)
                fig.suptitle('component ' + str(component_num) + ', strength = ' + str(float(strength)))
                plt.subplots_adjust(top=1.4)
                plt.savefig(folder + task_name + '_' + tensor_name + '_component_' + str(component_num) + '.png', bbox_inches='tight')
                plt.close()


print('done')
