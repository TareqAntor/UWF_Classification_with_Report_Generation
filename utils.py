# Printing out all outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
# PyTorch
import torch
from torchvision import transforms, models
from torch import optim, cuda, tensor
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
# warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# Data science tools
import numpy as np
import pandas as pd
import os
from skimage import io
# Image manipulations
from PIL import Image
# Timing utility
from timeit import default_timer as timer
# Visualizations
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
from tqdm import tqdm
# Metrics
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    roc_auc_score, multilabel_confusion_matrix,
    cohen_kappa_score
)


# ============================================================
#  DATASET & LABEL UTILITIES  (unchanged)
# ============================================================

class MyData(Dataset):

    def __init__(self, root_dir, categories, img_names, target,
                 my_transforms, return_path, ONN, mean, std):
        self.root_dir      = root_dir
        self.categories    = categories
        self.img_names     = img_names
        self.target        = target
        self.my_transforms = my_transforms
        self.return_path   = return_path
        self.ONN           = ONN
        self.mean          = mean
        self.std           = std
        if self.ONN:
            self.mean = torch.tensor(mean)
            self.std  = torch.tensor(std)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        y     = self.target[index].squeeze()
        label = y.item()
        x = io.imread(os.path.join(
            self.root_dir,
            self.categories[label] + '/' + self.img_names[index]
        ))
        x = self.my_transforms(x)
        if self.ONN:
            x = 2.0 * x - 1
        if self.return_path:
            return x, y, self.categories[label] + '/' + self.img_names[index]
        else:
            return x, y


def Createlabels(datadir):
    categories   = []
    n_Class      = []
    img_names    = []
    labels       = []
    i            = 0
    class_to_idx = {}
    idx_to_class = {}
    for d in os.listdir(datadir):
        class_to_idx[d] = i
        idx_to_class[i] = d
        categories.append(d)
        temp   = os.listdir(datadir + d)
        img_names.extend(temp)
        n_temp = len(temp)
        if i == 0:
            labels = np.zeros((n_temp, 1))
        else:
            labels = np.concatenate((labels, i * np.ones((n_temp, 1))))
        i += 1
        n_Class.append(n_temp)
    return categories, n_Class, img_names, labels, class_to_idx, idx_to_class


def Retrievelabel(datadir, categories):
    n_Class   = []
    img_names = []
    labels    = []
    for i, d in enumerate(categories):
        temp   = os.listdir(datadir + d)
        img_names.extend(temp)
        n_temp = len(temp)
        if i == 0:
            labels = np.zeros((n_temp, 1))
        else:
            labels = np.concatenate((labels, i * np.ones((n_temp, 1))))
        n_Class.append(n_temp)
    return n_Class, img_names, labels


def to_one_hot(y, n_dims=None):
    y_tensor  = y.data
    y_tensor  = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims    = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot


# ============================================================
#  METRICS — all 7 columns from the paper table
#  Accuracy | AUC | Precision | Sensitivity | F1 | Specificity | Kappa
# ============================================================

def compute_all_metrics(all_targets, pred_label, pred_probs, categories):
    """
    Compute the 7 metrics reported in the paper table.

    Parameters
    ----------
    all_targets : np.ndarray (N,)    integer ground-truth labels
    pred_label  : np.ndarray (N,)    integer predicted labels
    pred_probs  : np.ndarray (N, C)  softmax probabilities per class
    categories  : list[str]          class names ordered by label index

    Returns
    -------
    metrics   : dict  scalar macro-average values for every metric
    per_class : dict  per-class breakdown keyed by class name
    """
    metrics = {}

    # 1. Accuracy
    metrics['accuracy'] = round(
        accuracy_score(all_targets, pred_label) * 100, 2)

    # 2. AUC-ROC (One-vs-Rest, macro average)
    try:
        metrics['auc'] = round(
            roc_auc_score(
                all_targets, pred_probs,
                multi_class='ovr', average='macro'
            ) * 100, 2)
    except ValueError as e:
        metrics['auc'] = float('nan')
        print(f'  [AUC warning] {e}')

    # 3. Precision (macro)
    metrics['precision'] = round(
        precision_score(
            all_targets, pred_label,
            average='macro', zero_division=0
        ) * 100, 2)

    # 4 & 6. Sensitivity + Specificity — derived from per-class confusion matrices
    cm_pc = multilabel_confusion_matrix(all_targets, pred_label)
    # cm_pc[i] layout:  [[TN, FP],
    #                     [FN, TP]]

    per_class        = {}
    sensitivity_list = []
    specificity_list = []

    for i, cls in enumerate(categories):
        tn = cm_pc[i][0][0];  fp = cm_pc[i][0][1]
        fn = cm_pc[i][1][0];  tp = cm_pc[i][1][1]

        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        f1c  = 2 * prec * sens / (prec + sens + 1e-8)

        per_class[cls] = {
            'TP':          int(tp),
            'FP':          int(fp),
            'FN':          int(fn),
            'TN':          int(tn),
            'sensitivity': round(sens * 100, 2),
            'specificity': round(spec * 100, 2),
            'precision':   round(prec * 100, 2),
            'f1':          round(f1c  * 100, 2),
        }
        sensitivity_list.append(sens)
        specificity_list.append(spec)

    metrics['sensitivity'] = round(np.mean(sensitivity_list) * 100, 2)
    metrics['specificity'] = round(np.mean(specificity_list) * 100, 2)

    # 5. F1-score (macro)
    metrics['f1_macro'] = round(
        f1_score(all_targets, pred_label,
                 average='macro', zero_division=0) * 100, 2)

    # 7. Cohen's Kappa
    metrics['kappa'] = round(
        cohen_kappa_score(all_targets, pred_label) * 100, 2)

    return metrics, per_class


def print_metrics(metrics, per_class, categories, fold_idx=None):
    """Print metrics in paper-table format for a single fold or test set."""
    tag = f'FOLD {fold_idx}' if fold_idx is not None else 'TEST'
    w   = 65
    print('\n' + '=' * w)
    print(f'  {tag}  —  Results (macro averages, matching paper table)')
    print('=' * w)
    rows = [
        ('Accuracy (%)',    metrics['accuracy']),
        ('AUC (%)',         metrics['auc']),
        ('Precision (%)',   metrics['precision']),
        ('Sensitivity (%)', metrics['sensitivity']),
        ('F1-score (%)',    metrics['f1_macro']),
        ('Specificity (%)', metrics['specificity']),
        ('Kappa (%)',       metrics['kappa']),
    ]
    for label, val in rows:
        val_str = f'{val:.2f}' if not np.isnan(val) else 'N/A'
        print(f'  {label:<22} {val_str:>8}')

    print(f'\n  {"Per-Class Breakdown":}')
    hdr = f"  {'Class':<10} {'Sens%':>8} {'Spec%':>8} {'Prec%':>8} {'F1%':>8}  {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}"
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for cls in categories:
        pc = per_class[cls]
        print(
            f"  {cls:<10} {pc['sensitivity']:>8.2f} {pc['specificity']:>8.2f} "
            f"{pc['precision']:>8.2f} {pc['f1']:>8.2f}  "
            f"{pc['TP']:>4} {pc['FP']:>4} {pc['FN']:>4} {pc['TN']:>4}"
        )
    print('=' * w)


def summarize_folds(all_fold_metrics, categories):
    """
    Print mean ± std across all folds — paper table column order.

    Parameters
    ----------
    all_fold_metrics : list[dict]  one metrics dict per fold
    categories       : list[str]
    """
    key_label_pairs = [
        ('accuracy',    'Accuracy (%)'),
        ('auc',         'AUC (%)'),
        ('precision',   'Precision (%)'),
        ('sensitivity', 'Sensitivity (%)'),
        ('f1_macro',    'F1-score (%)'),
        ('specificity', 'Specificity (%)'),
        ('kappa',       'Kappa (%)'),
    ]
    w = 65
    print('\n' + '=' * w)
    print('  CROSS-VALIDATION SUMMARY  —  mean ± std across all folds')
    print('  (paper table column order)')
    print('=' * w)
    print(f"  {'Metric':<22} {'Mean':>8}   {'Std':>6}")
    print('  ' + '-' * 42)
    for key, lbl in key_label_pairs:
        vals = [m[key] for m in all_fold_metrics
                if not np.isnan(m.get(key, float('nan')))]
        print(f"  {lbl:<22} {np.mean(vals):>8.2f}%  ±{np.std(vals):>5.2f}%")
    print('=' * w)


# ============================================================
#  TRAINING LOOP  (unchanged from your original)
# ============================================================

def train(model_to_load,
          model,
          stop_criteria,
          criterion,
          optimizer,
          scheduler,
          train_loader,
          valid_loader,
          test_loader,
          save_file_name,
          train_on_gpu,
          history=[],
          max_epochs_stop=5,
          n_epochs=30,
          print_every=2):
    """Train a PyTorch Model"""

    epochs_no_improve = 0
    valid_loss_min    = np.inf
    valid_best_acc    = 0
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    for epoch in range(n_epochs):
        train_loss = valid_loss = test_loss = 0.0
        train_acc  = valid_acc  = test_acc  = 0
        model.train()

        for ii, (data, target) in tqdm(enumerate(train_loader),
                                       total=len(train_loader), leave=False):
            if train_on_gpu:
                data, target = (data.to('cuda', non_blocking=True),
                                target.to('cuda', non_blocking=True))
            optimizer.zero_grad()
            output = model(data)
            if model_to_load == 'inception_v3':
                output = output[0]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            train_acc += accuracy.item() * data.size(0)
            del output, data, target, loss, accuracy, pred, correct_tensor

        else:
            model.epochs += 1
            with torch.no_grad():
                model.eval()

                for data, target in tqdm(valid_loader,
                                         total=len(valid_loader), leave=False):
                    if train_on_gpu:
                        data, target = (data.to('cuda', non_blocking=True),
                                        target.to('cuda', non_blocking=True))
                    output = model(data)
                    loss   = criterion(output, target)
                    valid_loss += loss.item() * data.size(0)
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                    valid_acc += accuracy.item() * data.size(0)

                for data, target, _ in tqdm(test_loader,
                                            total=len(test_loader), leave=False):
                    if train_on_gpu:
                        data, target = (data.to('cuda', non_blocking=True),
                                        target.to('cuda', non_blocking=True))
                    output = model(data)
                    loss   = criterion(output, target)
                    test_loss += loss.item() * data.size(0)
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                    test_acc += accuracy.item() * data.size(0)

                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)
                test_loss  = test_loss  / len(test_loader.dataset)
                scheduler.step(valid_loss)
                train_acc  = train_acc  / len(train_loader.dataset)
                valid_acc  = valid_acc  / len(valid_loader.dataset)
                test_acc   = test_acc   / len(test_loader.dataset)

                history.append([train_loss, valid_loss, test_loss,
                                 train_acc,  valid_acc,  test_acc])

                if (epoch + 1) % print_every == 0:
                    print(
                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} '
                        f'\tValidation Loss: {valid_loss:.4f} '
                        f'\tTest Loss: {test_loss:.4f}'
                    )
                    print(
                        f'\t\tTraining Accuracy: {100*train_acc:.2f}% '
                        f'\tValidation Accuracy: {100*valid_acc:.2f}% '
                        f'\tTest Accuracy: {100*test_acc:.2f}%'
                    )

                del output, data, target, loss, accuracy, pred, correct_tensor

                if stop_criteria == 'loss':
                    if valid_loss < valid_loss_min:
                        torch.save(model.state_dict(), save_file_name)
                        epochs_no_improve = 0
                        valid_loss_min    = valid_loss
                        valid_best_acc    = valid_acc
                        best_epoch        = epoch
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= max_epochs_stop:
                            print(
                                f'\nEarly Stopping! Total epochs: {epoch}. '
                                f'Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} '
                                f'and acc: {100*valid_best_acc:.2f}%'
                            )
                            total_time = timer() - overall_start
                            print(f'{total_time:.2f} total seconds elapsed. '
                                  f'{total_time/(epoch+1):.2f} seconds per epoch.')
                            model.load_state_dict(
                                torch.load(save_file_name, weights_only=False))
                            model.optimizer = optimizer
                            history = pd.DataFrame(
                                history,
                                columns=['train_loss', 'val_loss', 'test_loss',
                                         'train_acc',  'val_acc',  'test_acc'])
                            return model, history

                elif stop_criteria == 'accuracy':
                    if valid_acc > valid_best_acc:
                        torch.save(model.state_dict(), save_file_name)
                        epochs_no_improve = 0
                        valid_loss_min    = valid_loss
                        valid_best_acc    = valid_acc
                        best_epoch        = epoch
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= max_epochs_stop:
                            print(
                                f'\nEarly Stopping! Total epochs: {epoch}. '
                                f'Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} '
                                f'and acc: {100*valid_best_acc:.2f}%'
                            )
                            total_time = timer() - overall_start
                            print(f'{total_time:.2f} total seconds elapsed. '
                                  f'{total_time/(epoch+1):.2f} seconds per epoch.')
                            model.load_state_dict(
                                torch.load(save_file_name, weights_only=False))
                            model.optimizer = optimizer
                            history = pd.DataFrame(
                                history,
                                columns=['train_loss', 'val_loss', 'test_loss',
                                         'train_acc',  'val_acc',  'test_acc'])
                            return model, history

    model.load_state_dict(torch.load(save_file_name, weights_only=False))
    model.optimizer = optimizer
    total_time = timer() - overall_start
    print(f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} '
          f'and acc: {100*valid_best_acc:.2f}%')
    print(f'{total_time:.2f} total seconds elapsed. '
          f'{total_time/epoch:.2f} seconds per epoch.')
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'val_loss', 'test_loss',
                 'train_acc',  'val_acc',  'test_acc'])
    return model, history
