import matplotlib.pyplot as plt
import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)

class Logger:
    """
    This class contains functionalities relating to the recording of model outputs, postprocessing,
    selection of most frequently sampled/highest scoring solutions, accuracy computations, and more.
    """
    ema_decay = 0.97

    def __init__(self, task):
        self.task = task
        self.KL_curves = {}
        self.total_KL_curve = []
        self.reconstruction_error_curve = []
        self.loss_curve = []

        n_test, n_colors, n_x, n_y = task.n_test, task.n_colors, task.n_x, task.n_y
        shape = (n_test, n_colors + 1, n_x, n_y)

        self.current_logits = torch.zeros(shape)
        self.current_x_mask = torch.zeros((n_test, n_x))
        self.current_y_mask = torch.zeros((n_test, n_y))

        self.ema_logits = torch.zeros(shape)
        self.ema_x_mask = torch.zeros((n_test, n_x))
        self.ema_y_mask = torch.zeros((n_test, n_y))

        self.solution_hashes_count = {}
        self.solution_most_frequent = None
        self.solution_second_most_frequent = None

        self.solution_contributions_log = []
        self.solution_picks_history = []

    def log(self, train_step, logits, x_mask, y_mask, KL_amounts, KL_names, total_KL, reconstruction_error, loss):
        """Logs training progress and tracks solutions from one forward pass."""
        if KL_amounts is not None and KL_names is not None:
            if train_step == 0:
                self.KL_curves = {KL_name: [] for KL_name in KL_names}

            for KL_amount, KL_name in zip(KL_amounts, KL_names):
                self.KL_curves[KL_name].append(float(KL_amount.detach().sum().cpu().to(torch.float32).numpy()))

            self.total_KL_curve.append(float(total_KL.detach().cpu().to(torch.float32).numpy()))
        
        if isinstance(reconstruction_error, torch.Tensor):
            self.reconstruction_error_curve.append(float(reconstruction_error.detach().cpu().to(torch.float32).numpy()))
        else:
            self.reconstruction_error_curve.append(reconstruction_error)
        self.loss_curve.append(float(loss.detach().cpu().to(torch.float32).numpy()))

        self._track_solution(
            train_step,
            logits.detach() if logits is not None else None,
            x_mask.detach() if x_mask is not None else None,
            y_mask.detach() if y_mask is not None else None
        )

    def _track_solution(self, train_step, logits, x_mask, y_mask):
        """Postprocess and score solutions and keep track of the top two solutions with highest scores."""
        
        
        
        # Check if we have None masks (indicating token-based model)
        if x_mask is None or y_mask is None:
            # For token-based models, store the logits directly for visualization
            if logits is not None:
                self.current_logits = logits
                # Use exponential moving average for token logits as well
                if not hasattr(self, '_token_ema_initialized'):
                    self.ema_logits = logits.clone()
                    self._token_ema_initialized = True
                else:
                    self.ema_logits = self.ema_decay * self.ema_logits + (1 - self.ema_decay) * logits
            return
        
        if logits is not None:
            if logits.dim() == 5 and logits.shape[-1] > 1:
                self.current_logits = logits[self.task.n_train:, :, :, :, 1]  # example, color, x, y
            else:
                self.current_logits = logits[self.task.n_train:]
        if x_mask is not None:
            if x_mask.dim() == 3 and x_mask.shape[-1] > 1:
                self.current_x_mask = x_mask[self.task.n_train:, :, 1]  # example, x
            else:
                self.current_x_mask = x_mask[self.task.n_train:]
        if y_mask is not None:
            if y_mask.dim() == 3 and y_mask.shape[-1] > 1:
                self.current_y_mask = y_mask[self.task.n_train:, :, 1]  # example, y
            else:
                self.current_y_mask = y_mask[self.task.n_train:]

        self.ema_logits = self.ema_decay * self.ema_logits + (1 - self.ema_decay) * self.current_logits
        self.ema_x_mask = self.ema_decay * self.ema_x_mask + (1 - self.ema_decay) * self.current_x_mask
        self.ema_y_mask = self.ema_decay * self.ema_y_mask + (1 - self.ema_decay) * self.current_y_mask

        solution_contributions = []
        for logits, x_mask_set, y_mask_set in [  # Add two potential solutions: sample and mean.
            (self.current_logits, self.current_x_mask, self.current_y_mask),
            (self.ema_logits, self.ema_x_mask, self.ema_y_mask)
        ]:

            # Get the solution and the score.
            solution, uncertainty = self._postprocess_solution(logits, x_mask_set, y_mask_set)
            hashed_solution = hash(solution)
            score = -10*uncertainty
            if train_step < 150:
                score = score - 10
            if logits is self.ema_logits:
                score = score - 4

            # Accumulate scores for solutions.
            solution_contributions.append((hashed_solution, score))
            self.solution_hashes_count[hashed_solution] = float(np.logaddexp(
                self.solution_hashes_count.get(hashed_solution, -np.inf), score))

            self._update_most_frequent_solutions(hashed_solution, solution)

        self.solution_contributions_log.append(solution_contributions)
        self.solution_picks_history.append([hash(sol) for sol in [
            self.solution_most_frequent, self.solution_second_most_frequent]])

    def _update_most_frequent_solutions(self, hashed, solution):
        """Keeps track of the top two solutions with highest scores."""
        if self.solution_most_frequent is None:
            self.solution_most_frequent = solution
        if self.solution_second_most_frequent is None:
            self.solution_second_most_frequent = solution

        if hashed != hash(self.solution_most_frequent):
            if self.solution_hashes_count[hashed] >= self.solution_hashes_count.get(
                    hash(self.solution_second_most_frequent), -np.inf):
                self.solution_second_most_frequent = solution
                if self.solution_hashes_count[hashed] >= self.solution_hashes_count.get(
                        hash(self.solution_most_frequent), -np.inf):
                    self.solution_second_most_frequent = self.solution_most_frequent
                    self.solution_most_frequent = solution

    def best_crop(self, prediction, x_mask, x_length, y_mask, y_length):
        x_start, x_end = self._best_slice_point(x_mask, x_length)
        y_start, y_end = self._best_slice_point(y_mask, y_length)
        return prediction[..., x_start:x_end, y_start:y_end]

    def _best_slice_point(self, mask, length):
        if self.task.in_out_same_size or self.task.all_out_same_size:
            search_lengths = [length]
        else:
            search_lengths = list(range(1, mask.shape[0]+1))
        max_logprob, best_slice_start, best_slice_end = None, None, None

        for length in search_lengths:
            logprobs = torch.stack([
                -torch.sum(mask[:offset]) + torch.sum(mask[offset:offset + length]) - torch.sum(mask[offset + length:])
                for offset in range(mask.shape[0] - length + 1)
            ])
            if max_logprob is None or torch.max(logprobs) > max_logprob:
                max_logprob = torch.max(logprobs)
                best_slice_start = torch.argmax(logprobs).item()
                best_slice_end = best_slice_start + length

        return best_slice_start, best_slice_end

    def _postprocess_solution(self, prediction, x_mask, y_mask):  # prediction must be example, color, x, y
        """Postprocess a solution and compute some variables that are used to calculate the score."""
        colors = torch.argmax(prediction, dim=1)  # example, x, y
        uncertainties = torch.logsumexp(prediction, dim=1) - torch.amax(prediction, dim=1)  # example, x, y
        solution_slices, uncertainty_values = [], []  # example, x, y; example

        for example_num in range(self.task.n_test):
            x_length = None
            y_length = None
            if self.task.in_out_same_size or self.task.all_out_same_size:
                x_length = self.task.shapes[self.task.n_train+example_num][1][0]
                y_length = self.task.shapes[self.task.n_train+example_num][1][1]
            solution_slice = self.best_crop(colors[example_num],
                                            x_mask[example_num],
                                            x_length,
                                            y_mask[example_num],
                                            y_length)  # x, y
            uncertainty_slice = self.best_crop(uncertainties[example_num],
                                               x_mask[example_num],
                                               x_length,
                                               y_mask[example_num],
                                               y_length)  # x, y

            solution_slices.append(solution_slice.cpu().to(torch.float32).numpy().tolist())
            uncertainty_values.append(float(np.mean(uncertainty_slice.cpu().to(torch.float32).numpy())))

        for example in solution_slices:
            for row in example:
                for i, val in enumerate(row):
                    row[i] = self.task.colors[int(val)]

        solution_slices = tuple(tuple(tuple(row) for row in example) for example in solution_slices)
        return solution_slices, np.mean(uncertainty_values)


def save_predictions(loggers, fname='predictions.npz'):
    """Saves solution score contributions and history of chosen solutions."""
    np.savez(fname,
             solution_contribution_logs=[logger.solution_contributions_log for logger in loggers],
             solution_picks_histories=[logger.solution_picks_history for logger in loggers])


def plot_accuracy(true_solution_hashes, fname='predictions.npz'):
    """Plots accuracy curve over training iterations."""
    stored_data = np.load(fname, allow_pickle=True)
    solution_picks_histories = stored_data['solution_picks_histories']

    n_tasks = len(solution_picks_histories)
    n_iterations = len(solution_picks_histories[0])

    correct = np.array([[
        int(any(hash_ == true_solution_hashes[task_num] for hash_ in solution_pair))
        for solution_pair in task_history
    ] for task_num, task_history in enumerate(solution_picks_histories)])

    accuracy_curve = correct.mean(axis=0)

    plt.figure()
    plt.plot(np.arange(n_iterations), accuracy_curve, 'k-')
    plt.savefig('accuracy_curve.pdf', bbox_inches='tight')
    plt.close()
