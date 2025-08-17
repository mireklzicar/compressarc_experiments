import argparse
import os
import pickle
import csv
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
from compression_vitarc import CompressionViTARC
from models.vit_arc.vitarc_with_gumbel import ViTARCWithGumbel
from models.vit_arc.train_step import compute_loss


"""
This file allows you to train one model on one task, and see plots of what
the process and end result looks like. You can input the training split and
the task code, and it will:
- Train a model for 200 steps,
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
torch.set_default_device('cuda')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='training', help='data split to use')
    # Some interesting tasks: 272f95fa, 6d75e8bb, 6cdd2623, 41e4d17e, 2bee17df
    # 228f6490, 508bd3b6, 2281f1f4, ecdecbb3
    parser.add_argument('--task_name', type=str, default='272f95fa', help='name of the task to run')
    parser.add_argument('--model_name', type=str, default='compressarc_baseline', help='name of the model to use')
    parser.add_argument('--use_gumbel_softmax', action='store_true', help='use gumbel softmax for op selection')
    parser.add_argument('--temperature', type=float, default=1.0, help='gumbel softmax temperature')
    args = parser.parse_args()
    split = args.split
    task_name = args.task_name
    model_name = args.model_name
    folder = f'outputs/{task_name}/{model_name}/'
    print('Performing a training run on task', task_name,
          'and placing the results in', folder)
    os.makedirs(folder, exist_ok=True)

    # Preprocess the task, set up the training
    task = preprocessing.preprocess_tasks(split, [task_name])[0]
    if model_name == 'differentiable_dsl':
        # For the differentiable_dsl model, we need to make sure that the ARCCompressor
        # is prepared to handle the dsl features.
        # For now, we will just pass the task, and the ARCCompressor will handle the rest.
        model = arc_compressor.ARCCompressor(
            task,
            use_dsl=True,
            use_gumbel_softmax=args.use_gumbel_softmax,
            temperature=args.temperature,
        )
    elif model_name == 'compression_vitarc':
        model = CompressionViTARC(task)
    elif model_name == 'compression_vitarc_gumbel':
        # Create the base CompressionViTARC model first
        base_model = CompressionViTARC(task)
        # Then wrap it with ViTARCWithGumbel
        # T5-small has d_model=512, not 768
        model = ViTARCWithGumbel(
            encoder=base_model.model.encoder,
            decoder=base_model.model.decoder,
            lm_head=base_model.model.lm_head,  # Add the language modeling head
            embed_dim=512,  # Match T5-small embedding dimension
            codebook_size=512,
            temp_init=1.0,
            temp_min=0.2,
            anneal_rate=1e-6,  # Much gentler annealing
            use_straight_through=True
        )
    else:
        model = arc_compressor.ARCCompressor(task, use_dsl=False)

    if model_name in ['compression_vitarc', 'compression_vitarc_gumbel']:
        n_iterations = 10000
        checkpoint_steps = 250
        if model_name == 'compression_vitarc':
            optimizer = torch.optim.Adam(model.model.parameters(), lr=0.0001, betas=(0.5, 0.9))
        else:  # compression_vitarc_gumbel
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.9))
    else:
        n_iterations = 200
        checkpoint_steps = 25
        optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
    scaler = torch.amp.GradScaler('cuda')
    train_history_logger = solution_selection.Logger(task)
    visualization.plot_problem(train_history_logger)

    for train_step in tqdm(range(n_iterations)):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            if model_name == 'compression_vitarc_gumbel':
                # Special handling for Gumbel model - use same approach as regular compression_vitarc
                optimizer.zero_grad()
                
                # Get tokenizer and prepare batch like regular compression_vitarc
                from tokenization.arc_tokenizer import get_or_build_arc_tokenizer
                tokenizer = get_or_build_arc_tokenizer()
                batch = train.prepare_vitarc_batch(task, tokenizer)
                
                # Move tensors to the correct device
                input_ids = batch['input_ids'].to(torch.get_default_device())
                attention_mask = batch['attention_mask'].to(torch.get_default_device())
                labels = batch['labels'].to(torch.get_default_device())
                
                # Forward pass through the gumbel model
                outputs_all, aux_all = model(input_ids, return_aux=True)
                
                # Extract training outputs for loss computation
                outputs_train = outputs_all[:task.n_train]
                labels_train = labels[:task.n_train]
                aux_train = {}
                for k, v in aux_all.items():
                    if isinstance(v, torch.Tensor) and len(v.shape) > 0 and v.shape[0] == outputs_all.shape[0]:
                        aux_train[k] = v[:task.n_train]
                    else:
                        aux_train[k] = v
                
                # Compute loss using only the training examples
                loss, metrics = compute_loss(outputs_train, labels_train, aux_train, w_entropy=0.0)
                
                # Log the model outputs and loss for visualization
                train_history_logger.log(
                    train_step=train_step,
                    logits=outputs_all,  # Pass ALL outputs (train + test) for token-based visualization
                    x_mask=None,     # Token-based models don't use spatial masks
                    y_mask=None,     # Token-based models don't use spatial masks
                    KL_amounts=[metrics["kl_uniform"]],
                    KL_names=["kl_uniform"],
                    total_KL=metrics["kl_uniform"],
                    reconstruction_error=metrics["loss_rec"],
                    loss=loss
                )
            else:
                loss = train.take_step(task, model, optimizer, train_step, train_history_logger)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Plot solutions and save metrics every checkpoint_steps steps
        if (train_step+1) % checkpoint_steps == 0:
            # Save model
            if hasattr(model, 'weights_list'):
                torch.save(model.weights_list, folder + f'model.pt')
            elif model_name in ['compression_vitarc', 'compression_vitarc_gumbel']:
                torch.save(model.state_dict(), folder + f'model.pt')

            # Save metrics
            metrics_path = folder + 'metrics.csv'
            write_header = not os.path.exists(metrics_path)
            with open(metrics_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    headers = ['step', 'loss', 'reconstruction_error', 'total_kl']
                    # Dynamically add KL curve names to header
                    headers.extend(train_history_logger.KL_curves.keys())
                    writer.writerow(headers)
                
                row = [
                    train_step + 1,
                    train_history_logger.loss_curve[-1],
                    train_history_logger.reconstruction_error_curve[-1],
                    train_history_logger.total_KL_curve[-1]
                ]
                # Add latest KL curve values to row
                for kl_curve in train_history_logger.KL_curves.values():
                    row.append(kl_curve[-1])
                writer.writerow(row)

            visualization.plot_solution(train_history_logger, model,
                fname=folder + task_name + '_at_' + str(train_step+1) + ' steps.png')
            visualization.plot_solution(train_history_logger, model,
                fname=folder + task_name + '_at_' + str(train_step+1) + ' steps.pdf')
    
    torch.cuda.empty_cache()

print('done')