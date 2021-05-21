import numpy as np
import torch
from pathlib import Path


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False, delta=0, save_path='checkpoint.pt', reverse=True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.Inf
        self.delta = delta
        self.save_path = save_path
        self.reverse = reverse
        self.__check_path__()

    def __check_path__(self):
        parent_dir = Path(self.save_path).parent
        if not parent_dir.exists(): parent_dir.mkdir()

    def __call__(self, score, model):

        if self.reverse:
            score = -score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'score ({self.val_score_min:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_score_min = score
