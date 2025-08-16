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
    else:
        model = arc_compressor.ARCCompressor(task, use_dsl=False)

    if model_name == 'compression_vitarc':
        n_iterations = 10000
        checkpoint_steps = 250
        optimizer = torch.optim.Adam(model.model.parameters(), lr=0.0001, betas=(0.5, 0.9))
    else:
        n_iterations = 200
        checkpoint_steps = 50
        optimizer = torch.optim.Adam(model.model.parameters(), lr=0.01, betas=(0.5, 0.9))
    scaler = torch.amp.GradScaler('cuda')
    train_history_logger = solution_selection.Logger(task)
    visualization.plot_problem(train_history_logger)

    for train_step in tqdm(range(n_iterations)):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = train.take_step(task, model, optimizer, train_step, train_history_logger)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Plot solutions and save metrics every checkpoint_steps steps
        if (train_step+1) % checkpoint_steps == 0:
            # Save model
            if hasattr(model, 'weights_list'):
                torch.save(model.weights_list, folder + f'model.pt')

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