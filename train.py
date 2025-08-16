import time
import sys
import os

import numpy as np
import torch

import preprocessing
import arc_compressor
import initializers
import multitensor_systems
import layers
import solution_selection
import visualization
from compression_vitarc import CompressionViTARC

# Add the local tokenizer path
from tokenization.arc_tokenizer import get_or_build_arc_tokenizer


"""
This file trains a model for every ARC-AGI task in a split.
"""

np.random.seed(0)
torch.manual_seed(0)


def arc_grid_to_tokens(grid):
    """
    Convert an ARC grid (2D numpy array) to a tokenized string.
    Each cell value (0-9) becomes <arc_0> through <arc_9>.
    Rows are separated by <arc_nl>.
    """
    token_strings = []
    for row in grid:
        row_tokens = [f"<arc_{cell}>" for cell in row]
        token_strings.append("".join(row_tokens))
    return "<arc_nl>".join(token_strings)

def prepare_vitarc_batch(task, tokenizer):
    """
    Prepare a batch for CompressionViTARC model from task data.
    Converts ARC grids to tokenized sequences.
    """
    input_texts = []
    target_texts = []
    
    # For training examples, use input->output pairs
    for i in range(task.n_train):
        input_grid = task.unprocessed_problem['train'][i]['input']
        output_grid = task.unprocessed_problem['train'][i]['output']
        
        input_text = arc_grid_to_tokens(np.array(input_grid))
        output_text = arc_grid_to_tokens(np.array(output_grid))
        
        input_texts.append(input_text)
        target_texts.append(output_text)
    
    if input_texts and target_texts:
        # Tokenize all inputs and targets
        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        targets = tokenizer(
            target_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        batch = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': targets['input_ids']
        }
        return batch
    
    # Fallback to dummy data if no training examples
    batch_size = 1
    seq_length = 32
    vocab_size = tokenizer.vocab_size
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    labels = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def mask_select_logprobs(mask, length):
    """
    Figure out the unnormalized log probability of taking each slice given the output mask.
    """
    logprobs = []
    for offset in range(mask.shape[0]-length+1):
        logprob = -torch.sum(mask[:offset])
        logprob = logprob + torch.sum(mask[offset:offset+length])
        logprob = logprob - torch.sum(mask[offset+length:])
        logprobs.append(logprob)
    logprobs = torch.stack(logprobs, dim=0)
    log_partition = torch.logsumexp(logprobs, dim=0)
    return log_partition, logprobs

def take_step(task, model, optimizer, train_step, train_history_logger):
    """
    Runs a forward pass of the model on the ARC-AGI task.
    Args:
        task (Task): The ARC-AGI task containing the problem.
        model (ArcCompressor): The VAE decoder model to run the forward pass with.
        optimizer (torch.optim.Optimizer): The optimizer used to take the step on the model weights.
        train_step (int): The training iteration number.
        train_history_logger (Logger): A logger object used for logging the forward pass outputs
                of the model, as well as accuracy and other things.
    """

    optimizer.zero_grad()
    if isinstance(model, CompressionViTARC):
        # Get the tokenizer for proper data preparation
        try:
            tokenizer = get_or_build_arc_tokenizer()
            batch = prepare_vitarc_batch(task, tokenizer)
            
            # Move tensors to the correct device
            input_ids = batch['input_ids'].to(torch.get_default_device())
            attention_mask = batch['attention_mask'].to(torch.get_default_device())
            labels = batch['labels'].to(torch.get_default_device())
            
            loss, logits = model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        except Exception as e:
            print(f"Warning: Could not load ARC tokenizer ({e}). Using dummy data.")
            # Fallback to dummy data
            batch_size = 1
            seq_length = 512
            vocab_size = model.model.config.vocab_size
            
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=torch.get_default_device())
            attention_mask = torch.ones((batch_size, seq_length), device=torch.get_default_device())
            labels = torch.randint(0, vocab_size, (batch_size, seq_length), device=torch.get_default_device())
            
            loss, logits = model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        # For CompressionViTARC, we calculate token accuracy as reconstruction error
        predicted_tokens = torch.argmax(logits, dim=-1)
        correct_predictions = (predicted_tokens == labels).sum()
        total_tokens = labels.numel()
        reconstruction_error = 1 - (correct_predictions / total_tokens)
        
        total_KL = torch.tensor(0.0) # KL divergence is not applicable here
        x_mask, y_mask, KL_amounts, KL_names = None, None, [], []
    else:
        logits, x_mask, y_mask, KL_amounts, KL_names, = model.forward()
        logits = torch.cat([torch.zeros_like(logits[:,:1,:,:]), logits], dim=1)  # add black color to logits

        # Compute the total KL loss
        total_KL = 0
        for KL_amount in KL_amounts:
            total_KL = total_KL + torch.sum(KL_amount)

        # Compute the reconstruction error
        reconstruction_error = 0
        for example_num in range(task.n_examples):  # sum over examples
            for in_out_mode in range(2):  # sum over in/out grid per example
                if example_num >= task.n_train and in_out_mode == 1:
                    continue

                # Determine whether the grid size is already known.
                # If not, there is an extra term in the reconstruction error, corresponding to
                # the probability of reconstructing the correct grid size.
                grid_size_uncertain = not (task.in_out_same_size or task.all_out_same_size and in_out_mode==1 or task.all_in_same_size and in_out_mode==0)
                if grid_size_uncertain:
                    coefficient = 0.01**max(0, 1-train_step/100)
                else:
                    coefficient = 1
                logits_slice = logits[example_num,:,:,:,in_out_mode]  # color, x, y
                problem_slice = task.problem[example_num,:,:,in_out_mode]  # x, y
                output_shape = task.shapes[example_num][in_out_mode]
                x_log_partition, x_logprobs = mask_select_logprobs(coefficient*x_mask[example_num,:,in_out_mode], output_shape[0])
                y_log_partition, y_logprobs = mask_select_logprobs(coefficient*y_mask[example_num,:,in_out_mode], output_shape[1])
                # Account for probability of getting right grid size, if grid size is not known
                if grid_size_uncertain:
                    x_log_partitions = []
                    y_log_partitions = []
                    for length in range(1, x_mask.shape[1]+1):
                        x_log_partitions.append(mask_select_logprobs(coefficient*x_mask[example_num,:,in_out_mode], length)[0])
                    for length in range(1, y_mask.shape[1]+1):
                        y_log_partitions.append(mask_select_logprobs(coefficient*y_mask[example_num,:,in_out_mode], length)[0])
                    x_log_partition = torch.logsumexp(torch.stack(x_log_partitions, dim=0), dim=0)
                    y_log_partition = torch.logsumexp(torch.stack(y_log_partitions, dim=0), dim=0)

                # Given that we have the correct grid size, get the reconstruction error of getting the colors right
                logprobs = [[] for x_offset in range(x_logprobs.shape[0])]  # x, y
                for x_offset in range(x_logprobs.shape[0]):
                    for y_offset in range(y_logprobs.shape[0]):
                        logprob = x_logprobs[x_offset] - x_log_partition + y_logprobs[y_offset] - y_log_partition  # given the correct grid size,
                        logits_crop = logits_slice[:,x_offset:x_offset+output_shape[0],y_offset:y_offset+output_shape[1]]  # c, x, y
                        target_crop = problem_slice[:output_shape[0],:output_shape[1]]  # x, y
                        logprob = logprob - torch.nn.functional.cross_entropy(logits_crop[None,...], target_crop[None,...], reduction='sum')  # calculate the error for the colors.
                        logprobs[x_offset].append(logprob)
                logprobs = torch.stack([torch.stack(logprobs_, dim=0) for logprobs_ in logprobs], dim=0)  # x, y
                if grid_size_uncertain:
                    coefficient = 0.1**max(0, 1-train_step/100)
                else:
                    coefficient = 1
                logprob = torch.logsumexp(coefficient*logprobs, dim=(0,1))/coefficient  # Aggregate for all possible grid sizes
                reconstruction_error = reconstruction_error - logprob
        loss = total_KL + 10*reconstruction_error

    # Performance recording
    train_history_logger.log(train_step,
                             logits,
                             x_mask,
                             y_mask,
                             KL_amounts,
                             KL_names,
                             total_KL,
                             reconstruction_error,
                             loss)
    return loss


if __name__ == "__main__":
    start_time = time.time()

    task_nums = list(range(400))
    split = "training"  # "training", "evaluation, or "test"

    # Preprocess all tasks, make models, optimizers, and loggers. Make plots.
    tasks = preprocessing.preprocess_tasks(split, task_nums)
    models = []
    optimizers = []
    train_history_loggers = []
    for task in tasks:
        model = arc_compressor.ARCCompressor(task)
        models.append(model)
        optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
        optimizers.append(optimizer)
        train_history_logger = solution_selection.Logger(task)
        visualization.plot_problem(train_history_logger)
        train_history_loggers.append(train_history_logger)

    # Get the solution hashes so that we can check for correctness
    true_solution_hashes = [task.solution_hash for task in tasks]

    # Train the models one by one
    for i, (task, model, optimizer, train_history_logger) in enumerate(zip(tasks, models, optimizers, train_history_loggers)):
        n_iterations = 2000
        for train_step in range(n_iterations):
            take_step(task, model, optimizer, train_step, train_history_logger)
        visualization.plot_solution(train_history_logger)
        solution_selection.save_predictions(train_history_loggers[:i+1])
        solution_selection.plot_accuracy(true_solution_hashes)

    # Write down how long it all took
    with open('timing_result.txt', 'w') as f:
        f.write("Time elapsed in seconds: " + str(time.time() - start_time))
