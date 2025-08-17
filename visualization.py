import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from compression_vitarc import CompressionViTARC
from models.vit_arc.vitarc_with_gumbel import ViTARCWithGumbel
from tokenization.arc_tokenizer import get_or_build_arc_tokenizer


"""
This file trains a model for every ARC-AGI task in a split.
"""


def tokens_to_grid(tokens, tokenizer):
    """
    Convert a sequence of tokens back to a colored grid.
    Based on the reference implementation from vitarc_tetrominoes.
    """
    import re
    
    # Detokenize the sequence
    # Make sure tokens is a list of ints
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    
    # decode expects a list of ints, not a list of lists of ints
    if len(tokens) > 0 and isinstance(tokens[0], list):
        tokens = tokens[0]

    try:
        text = tokenizer.decode(tokens, skip_special_tokens=False)
        
        # Debug: #print first 200 characters of decoded text to see what we're working with
        #print(f"Debug - Decoded text (first 200 chars): {text[:200]}")
        
        # Clean up special tokens based on reference implementation
        cleaned_string = re.sub(r'<s>|</s>|<pad>', '', text).strip()
        
        # Check if we have newline tokens
        newline_count = cleaned_string.count('<arc_nl>')
        
        if newline_count > 0:
            # Normal case: grid has row separators
            rows = cleaned_string.split('<arc_nl>')
            
            grid = []
            for row_str in rows:
                if not row_str:
                    continue
                
                # Find all <arc_DIGIT> tokens in the row using regex
                found_tokens = re.findall(r'<arc_(\d+)>', row_str)
                if found_tokens:
                    row = [int(t) for t in found_tokens[:30]]  # Max 30 cols
                    grid.append(row)
                
                # Limit number of rows
                if len(grid) >= 30:
                    break
        else:
            # Fallback: flat sequence without newlines - try to reshape into 2D grid
            #print("Debug - No newline tokens found, attempting to reshape flat sequence")
            
            # Find all tokens in the entire string
            all_tokens = re.findall(r'<arc_(\d+)>', cleaned_string)
            
            if all_tokens:
                token_values = [int(t) for t in all_tokens]
                total_tokens = len(token_values)
                
                # Try to infer grid dimensions - prefer square-ish grids
                # Common ARC grid sizes: 3x3, 5x5, 10x10, etc.
                possible_dims = []
                for h in range(1, min(21, total_tokens + 1)):  # Max 20 rows
                    if total_tokens % h == 0:
                        w = total_tokens // h
                        if w <= 30:  # Max 30 cols
                            possible_dims.append((h, w))
                
                if possible_dims:
                    # Choose the most square-like dimensions
                    h, w = min(possible_dims, key=lambda x: abs(x[0] - x[1]))
                    #print(f"Debug - Reshaping {total_tokens} tokens to {h}x{w} grid")
                    
                    grid = []
                    for i in range(h):
                        start_idx = i * w
                        end_idx = start_idx + w
                        row = token_values[start_idx:end_idx]
                        grid.append(row)
                else:
                    # Can't reshape cleanly - create a single row (fallback)
                    #print(f"Debug - Can't reshape {total_tokens} tokens, using single row")
                    grid = [token_values[:30]]  # Max 30 cols
            else:
                grid = []
        
        if not grid:
            return np.zeros((3, 3, 3), dtype=np.uint8)  # Default small grid

        # Limit grid size to reasonable ARC dimensions
        grid = grid[:20]  # Max 20 rows
        
        # Ensure all rows have same length (pad with zeros) and limit width
        if grid:
            max_len = min(max(len(row) for row in grid), 20)  # Cap at 20 cols
        else:
            max_len = 3
        
        padded_grid = []
        for row in grid:
            padded_row = row[:max_len] + [0] * max(0, max_len - len(row))
            padded_grid.append(padded_row)

        # Convert to numpy array
        grid_arr = np.array(padded_grid, dtype=int)
        
        # Ensure values are in valid ARC color range (0-9)
        grid_arr = np.clip(grid_arr, 0, 9)
        
        # Crop the grid to remove exterior rows/columns of background color (0)
        grid_arr = crop_grid_numpy(grid_arr, background_color_idx=0)
        
        # Apply ARC color mapping: convert to one-hot then to RGB colors
        colored_grid = (np.arange(10) == grid_arr[..., None]).astype(np.float32)
        colored_grid = convert_color(colored_grid)
        
        return colored_grid
    
    except Exception as e:
        #print(f"Warning: Error in tokens_to_grid: {e}")
        # Return a small default grid on any error
        return np.zeros((3, 3, 3), dtype=np.uint8)

np.random.seed(0)
torch.manual_seed(0)


color_list = np.array([
    [0, 0, 0],  # black
    [30, 147, 255],  # blue
    [249, 60, 49],  # red
    [79, 204, 48],  # green
    [255, 220, 0],  # yellow
    [153, 153, 153],  # gray
    [229, 58, 163],  # magenta
    [255, 133, 27],  # orange
    [135, 216, 241],  # light blue
    [146, 18, 49],  # brown
])

def crop_grid_numpy(grid, background_color_idx=0):
    """Crops a numpy grid to remove exterior rows/columns of a specific background color."""
    if grid.size == 0:
        return grid

    # Find rows that are not all background color
    non_bg_rows = np.where(np.any(grid != background_color_idx, axis=1))[0]
    
    # Find columns that are not all background color
    non_bg_cols = np.where(np.any(grid != background_color_idx, axis=0))[0]
    
    # If the grid is all background or very small non-background content, don't crop aggressively
    if non_bg_rows.size == 0 or non_bg_cols.size == 0:
        # Return a reasonable sized crop instead of single pixel
        min_size = min(grid.shape[0], grid.shape[1], 5)  # At most 5x5, at least what we have
        return grid[:min_size, :min_size]
    
    # Add some padding around the content to avoid over-cropping
    row_start = max(0, non_bg_rows[0] - 1)
    row_end = min(grid.shape[0], non_bg_rows[-1] + 2)
    col_start = max(0, non_bg_cols[0] - 1)
    col_end = min(grid.shape[1], non_bg_cols[-1] + 2)
    
    # Crop the grid
    cropped_grid = grid[row_start:row_end, col_start:col_end]
    
    return cropped_grid

def convert_color(grid):  # grid dims must end in c
    return np.clip(np.matmul(grid, color_list), 0, 255).astype(np.uint8)

def plot_problem(logger):
    """
    Draw a plot of an ARC-AGI problem, and save it in plots/
    Args:
        logger (Logger): A logger object used to log model outputs for the ARC-AGI task.
    """

    # Put all the grids beside one another on one grid
    n_train = logger.task.n_train
    n_test = logger.task.n_test
    n_examples = logger.task.n_examples
    n_x = logger.task.n_x
    n_y = logger.task.n_y
    pixels = 255+np.zeros([n_train+n_test, 2*n_x+2, 2, 2*n_y+8, 3], dtype=np.uint8)
    for example_num in range(n_examples):
        if example_num < n_train:
            subsplit = 'train'
            subsplit_example_num = example_num
        else:
            subsplit = 'test'
            subsplit_example_num = example_num - n_train
        for mode_num, mode in enumerate(('input', 'output')):
            if subsplit == 'test' and mode == 'output':
                continue
            grid = np.array(logger.task.unprocessed_problem[subsplit][subsplit_example_num][mode])  # x, y
            grid = (np.arange(10)==grid[:,:,None]).astype(np.float32)  # x, y, c
            grid = convert_color(grid)  # x, y, c
            repeat_grid = np.repeat(grid, 2, axis=0)
            repeat_grid = np.repeat(repeat_grid, 2, axis=1)
            pixels[example_num,n_x+1-grid.shape[0]:n_x+1+grid.shape[0],mode_num,n_y+4-grid.shape[1]:n_y+4+grid.shape[1],:] = repeat_grid
    pixels = pixels.reshape([(n_train+n_test)*(2*n_x+2), 2*(2*n_y+8), 3])
    
    os.makedirs("plots/", exist_ok=True)

    # Plot the combined grid and make gray dividers between the grid cells, arrows, and a question mark for unsolved examples.
    fig, ax = plt.subplots()
    ax.imshow(pixels, aspect='equal', interpolation='none')
    for example_num in range(n_examples):
        for mode_num, mode in enumerate(('input', 'output')):
            if example_num < n_train:
                subsplit = 'train'
                subsplit_example_num = example_num
            else:
                subsplit = 'test'
                subsplit_example_num = example_num - n_train
            ax.arrow((2*n_y+8)-3-0.5, (2*n_x+2)*example_num+1+n_x-0.5, 6, 0, width=0.5, fc='k', ec='k', length_includes_head=True)
            if subsplit == 'test' and mode == 'output':
                ax.text((2*n_y+8)+4+n_y-0.5, (2*n_x+2)*example_num+1+n_x-0.5, '?', size='xx-large', ha='center', va='center')
                continue
            grid = np.array(logger.task.unprocessed_problem[subsplit][subsplit_example_num][mode])  # x, y
            for xline in range(grid.shape[0]+1):
                ax.plot(((2*n_y+8)*mode_num+4+n_y-grid.shape[1]-0.5, (2*n_y+8)*mode_num+4+n_y+grid.shape[1]-0.5),
                        ((2*n_x+2)*example_num+1+n_x-grid.shape[0]+2*xline-0.5,)*2,
                        color=(59/255, 59/255, 59/255),
                        linewidth=0.3)
            for yline in range(grid.shape[1]+1):
                ax.plot(((2*n_y+8)*mode_num+4+n_y-grid.shape[1]+2*yline-0.5,)*2,
                        ((2*n_x+2)*example_num+1+n_x-grid.shape[0]-0.5, (2*n_x+2)*example_num+1+n_x+grid.shape[0]-0.5),
                        color=(59/255, 59/255, 59/255),
                        linewidth=0.3)
    plt.axis('off')
    plt.savefig('plots/' + logger.task.task_name + '_problem.png', bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_solution(logger, model=None, fname=None):
    """
    Draw a plot of a model's solution to an ARC-AGI problem, and save it in plots/
    Draws four plots: A model output sample, the mean of samples, and the top two most common samples.
    Args:
        logger (Logger): A logger object used to log model outputs for the ARC-AGI task.
        model: The model object (needed to determine model type for proper visualization).
    """
    n_train = logger.task.n_train
    n_test = logger.task.n_test
    n_examples = logger.task.n_examples
    n_x = logger.task.n_x
    n_y = logger.task.n_y

    # Four plotted solutions
    if isinstance(model, (CompressionViTARC, ViTARCWithGumbel)):
        tokenizer = get_or_build_arc_tokenizer()
        
        # Predicted tokens from logits
        predicted_tokens_current = torch.argmax(logger.current_logits, dim=-1)
        predicted_tokens_ema = torch.argmax(logger.ema_logits, dim=-1)
        
        # Create solutions for each test example by converting tokens to grids
        current_solutions = []
        ema_solutions = []
        
        for test_idx in range(n_test):
            # The token sequences now include both training and test data.
            # Test examples start after the training examples.
            token_idx = n_train + test_idx
            
            current_grid = tokens_to_grid(predicted_tokens_current[token_idx], tokenizer)
            ema_grid = tokens_to_grid(predicted_tokens_ema[token_idx], tokenizer)
            
            current_solutions.append(current_grid)
            ema_solutions.append(ema_grid)

        solutions_list = [
            current_solutions,  # Now properly structured as array of grids by test example
            ema_solutions,      # Now properly structured as array of grids by test example
            logger.solution_most_frequent,
            logger.solution_second_most_frequent,
        ]
        
        masks_list = [None, None, None, None]
    else:
        solutions_list = [
                torch.softmax(logger.current_logits, dim=1).cpu().to(torch.float32).numpy(),
                torch.softmax(logger.ema_logits, dim=1).cpu().to(torch.float32).numpy(),
                logger.solution_most_frequent,
                logger.solution_second_most_frequent,
                ]
        masks_list = [
                (logger.current_x_mask, logger.current_y_mask),
                (logger.ema_x_mask, logger.ema_y_mask),
                None,
                None,
                ]
    solutions_labels = [
            'sample',
            'sample average',
            'guess 1',
            'guess 2',
            ]
    n_plotted_solutions = len(solutions_list)

    # Filter out None solutions and adjust accordingly
    valid_solutions = []
    valid_masks = []
    valid_labels = []
    for solution, masks, label in zip(solutions_list, masks_list, solutions_labels):
        if solution is not None:
            valid_solutions.append(solution)
            valid_masks.append(masks)
            valid_labels.append(label)
    
    n_plotted_solutions = len(valid_solutions)
    if n_plotted_solutions == 0:
        # No valid solutions to plot
        return

    # Put all the grids beside one another on one grid
    pixels = 255+np.zeros([n_test, 2*n_x+2, n_plotted_solutions, 2*n_y+8, 3], dtype=np.uint8)
    shapes = []
    for subsplit_example_num in range(n_test):
        subsplit = 'test'
        example_num = subsplit_example_num + n_train
        shapes.append([])

        for solution_num, (solution, masks, label) in enumerate(zip(valid_solutions, valid_masks, valid_labels)):
            if isinstance(model, (CompressionViTARC, ViTARCWithGumbel)):
                if 'guess' in label:
                    # Guess solutions already contain actual color values from task.colors
                    # We need to convert them back to indices and then to RGB colors
                    solution_grid = solution[subsplit_example_num]  # tuple of tuples (rows of color values)
                    
                    # Convert to numpy array
                    grid_array = np.array(solution_grid)
                    
                    # Map color values back to indices (0-9) for visualization
                    color_to_index = {color: idx for idx, color in enumerate(logger.task.colors)}
                    index_grid = np.zeros_like(grid_array, dtype=int)
                    
                    for i in range(grid_array.shape[0]):
                        for j in range(grid_array.shape[1]):
                            color_val = grid_array[i, j]
                            index_grid[i, j] = color_to_index.get(color_val, 0)  # Default to 0 if not found
                    
                    # Convert to one-hot encoding then to RGB colors
                    grid = (np.arange(10) == index_grid[..., None]).astype(np.float32)
                    grid = convert_color(grid)
                else:
                    grid = solution[subsplit_example_num]  # Now properly indexed by test example
                    
                # Ensure grid fits within visualization bounds for CompressionViTARC
                max_grid_x = 2 * n_x
                max_grid_y = 2 * n_y
                if grid.shape[0] > max_grid_x or grid.shape[1] > max_grid_y:
                    # Crop the grid to fit within bounds
                    crop_x = min(grid.shape[0], max_grid_x)
                    crop_y = min(grid.shape[1], max_grid_y)
                    grid = grid[:crop_x, :crop_y, :]
            else:
                grid = np.array(solution[subsplit_example_num])

                if 'sample' in label:
                    grid = np.einsum('dxy,dc->xyc', grid, color_list[logger.task.colors])  # x, y, c
                    if logger.task.in_out_same_size or logger.task.all_out_same_size:
                        x_length = logger.task.shapes[example_num][1][0]
                        y_length = logger.task.shapes[example_num][1][1]
                    else:
                        x_length = None
                        y_length = None
                    x_start, x_end = logger._best_slice_point(masks[0][subsplit_example_num,:], x_length)
                    y_start, y_end = logger._best_slice_point(masks[1][subsplit_example_num,:], y_length)
                    grid = grid[x_start:x_end,y_start:y_end,:]  # x, y, c
                    grid = np.clip(grid, 0, 255).astype(np.uint8)
                else:
                    grid = (np.arange(10)==grid[:,:,None]).astype(np.float32)  # x, y, c
                    grid = convert_color(grid)  # x, y, c

            shapes[subsplit_example_num].append((grid.shape[0], grid.shape[1]))
            repeat_grid = np.repeat(grid, 2, axis=0)
            repeat_grid = np.repeat(repeat_grid, 2, axis=1)
            pixels[subsplit_example_num,n_x+1-grid.shape[0]:n_x+1+grid.shape[0],solution_num,n_y+4-grid.shape[1]:n_y+4+grid.shape[1],:] = repeat_grid

    pixels = pixels.reshape([n_test*(2*n_x+2), n_plotted_solutions*(2*n_y+8), 3])
    
    # Plot the combined grid and make gray dividers between the grid cells, and labels.
    fig, ax = plt.subplots()
    ax.imshow(pixels, aspect='equal', interpolation='none')
    for subsplit_example_num in range(n_test):
        for solution_num in range(n_plotted_solutions):
            subsplit = 'test'
            # Now all solutions are properly structured as arrays indexed by test example
            grid = np.array(valid_solutions[solution_num][subsplit_example_num])  # x, y
            shape = shapes[subsplit_example_num][solution_num]
            for xline in range(shape[0]+1):
                ax.plot(((2*n_y+8)*solution_num+4+n_y-shape[1]-0.5, (2*n_y+8)*solution_num+4+n_y+shape[1]-0.5),
                        ((2*n_x+2)*subsplit_example_num+1+n_x-shape[0]+2*xline-0.5,)*2,
                        color=(59/255, 59/255, 59/255),
                        linewidth=0.3)
            for yline in range(shape[1]+1):
                ax.plot(((2*n_y+8)*solution_num+4+n_y-shape[1]+2*yline-0.5,)*2,
                        ((2*n_x+2)*subsplit_example_num+1+n_x-shape[0]-0.5, (2*n_x+2)*subsplit_example_num+1+n_x+shape[0]-0.5),
                        color=(59/255, 59/255, 59/255),
                        linewidth=0.3)
    for solution_num, solution_label in enumerate(valid_labels):
        ax.text((2*n_y+8)*solution_num+4+n_y-0.5, -3, solution_label, size='xx-small', ha='center', va='center')
    plt.axis('off')
    if fname is None:
        fname = 'plots/' + logger.task.task_name + '_solutions.pdf'
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close()


