import matplotlib
matplotlib.use('Agg')                          # ← MUST be before any other matplotlib import

# Printing out all outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
# PyTorch
import torch
from torchvision import transforms, models
from torch import optim, cuda, tensor
from torch.utils.data import DataLoader
import torch.nn as nn
# warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# Data science tools
import numpy as np
import os
from os import path
from importlib import import_module
# Visualizations
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
# customized functions
from utils import *
from models import *

from torch.serialization import SourceChangeWarning
for warning in [UserWarning, SourceChangeWarning, Warning]:
    warnings.filterwarnings("ignore", category=warning)

# Parse command line arguments
fname = "config.py"
configuration = import_module(fname.split(".")[0])
config = configuration.config


class EncoderModel(nn.Module):
    def __init__(self, old_model):
        super(EncoderModel, self).__init__()
        self.old_model = old_model
        self.new_model = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(512, class_num),   # ResNet18
            # nn.Linear(2048, class_num),  # ResNet50
            nn.LogSoftmax(dim=1))
        self.old_model = self.old_model.to('cuda')
        self.new_model = self.new_model.to('cuda')

    def forward(self, x):
        x = self.old_model(x)
        x = self.new_model(x[5])  # ResNet18
        return x


if __name__ == '__main__':

    ################## Network hyper-parameters
    parentdir        = config['parentdir']
    ImageNet         = config['ImageNet']
    q_order          = config['q_order']
    ONN              = config['ONN']
    input_ch         = config['input_ch']
    batch_size       = config['batch_size']
    input_mean       = config['input_mean']
    input_std        = config['input_std']
    optim_fc         = config['optim_fc']
    lr               = config['lr']
    stop_criteria    = config['stop_criteria']
    n_epochs         = config['n_epochs']
    epochs_patience  = config['epochs_patience']
    lr_factor        = config['lr_factor']
    max_epochs_stop  = config['max_epochs_stop']
    num_folds        = config['num_folds']
    Resize_h         = config['Resize_h']
    Resize_w         = config['Resize_w']
    load_model       = config['load_model']
    model_name       = config['model_name']
    model_to_load    = config['model_to_load']
    fold_to_run      = config['fold_to_run']
    encoder          = config['encoder']
    Results_path     = config['Results_path']
    save_path        = config['save_path']
    fold_to_run      = config['fold_to_run']
    RotaionDegree    = config['RotaionDegree']
    RHFlip           = config['RHFlip']
    P_padding        = config['P_padding']
    P_fill           = config['P_fill']
    P_padding_mode   = config['P_padding_mode'] = 'constant'
    dataset_location = config['dataset_location']
    ##################

    traindir = dataset_location + 'Data/Train/'
    testdir  = dataset_location + 'Data/Test/'
    valdir   = dataset_location + 'Data/Val/'

    # Create result directories
    for d in [Results_path, save_path]:
        if not path.exists(d):
            os.makedirs(d, exist_ok=True)

    # GPU setup
    train_on_gpu = cuda.is_available()
    print(f'Train on gpu: {train_on_gpu}')
    if train_on_gpu:
        gpu_count = cuda.device_count()
        print(f'{gpu_count} gpus detected.')
        multi_gpu = gpu_count > 1
    else:
        multi_gpu = False

    # Fold loop range
    if not fold_to_run:
        loop_start = 1
        loop_end   = num_folds + 1
    else:
        loop_start = fold_to_run[0]
        loop_end   = fold_to_run[1] + 1

    # ── CLASSES — must match your folder names exactly ──────────────
    # Createlabels() reads them from disk; order may vary by OS.
    # After the first fold we lock `categories` so all folds use the
    # same label-to-index mapping.
    locked_categories = None

    # Collect per-fold metric dicts for the final summary
    all_fold_metrics   = []
    all_fold_per_class = []   # ← per-class breakdown across folds

    # ==================================================================
    #  MAIN FOLD LOOP
    # ==================================================================
    for fold_idx in range(loop_start, loop_end):
        print('#############################################################')
        if fold_idx == loop_start:
            print('Training using ' + model_to_load + ' network')
        print(f'Started fold {fold_idx}')

        save_file_name  = save_path + '/' + model_name + f'_fold_{fold_idx}.pt'
        checkpoint_name = save_path + f'/checkpoint_{fold_idx}.pt'
        traindir_fold   = traindir + f'fold_{fold_idx}/'
        testdir_fold    = testdir  + f'fold_{fold_idx}/'
        valdir_fold     = valdir   + f'fold_{fold_idx}/'

        # ── Labels ──────────────────────────────────────────────────
        categories, n_Class_train, img_names_train, labels_train, \
            class_to_idx, idx_to_class = Createlabels(traindir_fold)

        # Lock category order after first fold so indices stay consistent
        if locked_categories is None:
            locked_categories = categories
        else:
            categories = locked_categories

        labels_train = torch.from_numpy(labels_train).to(torch.int64)
        class_num    = len(categories)

        _, n_Class_val,  img_names_val,  labels_val,  _, _ = Createlabels(valdir_fold)
        labels_val  = torch.from_numpy(labels_val).to(torch.int64)

        _, n_Class_test, img_names_test, labels_test, _, _ = Createlabels(testdir_fold)
        labels_test = torch.from_numpy(labels_test).to(torch.int64)

        # ── Transforms ──────────────────────────────────────────────
        if ONN:
            if input_ch == 3:
                my_transforms = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((Resize_h, Resize_w), interpolation=Image.BICUBIC),
                    transforms.ToTensor(),
                ])
                my_test_transforms = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((Resize_h, Resize_w), interpolation=Image.BICUBIC),
                    transforms.ToTensor(),
                ])
            elif input_ch == 1:
                my_transforms = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((Resize_h, Resize_w), interpolation=Image.BICUBIC),
                    transforms.ToTensor(),
                ])
                my_test_transforms = my_transforms
        else:
            if input_ch == 1 and len(input_mean) == 3:
                my_transforms = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize((Resize_h, Resize_w), interpolation=Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=input_mean, std=input_std),
                ])
            elif input_ch == 1:
                my_transforms = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((Resize_h, Resize_w), interpolation=Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=input_mean, std=input_std),
                ])
            else:
                my_transforms = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((Resize_h, Resize_w), interpolation=Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=input_mean, std=input_std),
                ])
            my_test_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((Resize_h, Resize_w), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=input_mean, std=input_std),
            ])

        # ── DataLoaders ─────────────────────────────────────────────
        train_ds = MyData(root_dir=traindir_fold, categories=categories,
                          img_names=img_names_train, target=labels_train,
                          my_transforms=my_transforms, return_path=False,
                          ONN=ONN, mean=input_mean, std=input_std)
        drop = (len(train_ds) % batch_size) != 0
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=1, drop_last=drop)

        val_ds = MyData(root_dir=valdir_fold, categories=categories,
                        img_names=img_names_val, target=labels_val,
                        my_transforms=my_test_transforms, return_path=False,
                        ONN=ONN, mean=input_mean, std=input_std)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            pin_memory=True, num_workers=1)

        test_ds = MyData(root_dir=testdir_fold, categories=categories,
                         img_names=img_names_test, target=labels_test,
                         my_transforms=my_test_transforms, return_path=True,
                         ONN=ONN, mean=input_mean, std=input_std)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             pin_memory=True, num_workers=1)

        del n_Class_train, img_names_train, labels_train
        del n_Class_val,   img_names_val,   labels_val

        # ── Model ───────────────────────────────────────────────────
        if load_model:
            checkpoint = torch.load(load_model)
            model = checkpoint['model']
            del checkpoint
            if encoder:
                model = EncoderModel(model.encoder)
                model = model.to('cuda')
        else:
            model = get_pretrained_model(parentdir, model_to_load, ImageNet,
                                         input_ch, class_num, train_on_gpu,
                                         multi_gpu, q_order)

        if next(model.parameters()).is_cuda:
            print('Model device: cuda')

        # ── Loss & Optimizer ────────────────────────────────────────
        criterion = nn.NLLLoss()

        if optim_fc == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr,
                                   betas=(0.9, 0.999), eps=1e-08,
                                   weight_decay=0.0001, amsgrad=False)
        elif optim_fc == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                  dampening=0, weight_decay=0.0001, nesterov=False)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=lr_factor, patience=epochs_patience,
            threshold=0.0001, threshold_mode='rel', cooldown=0,
            min_lr=0, eps=1e-08)

        # ── Training ────────────────────────────────────────────────
        model, history = train(
            model_to_load, model, stop_criteria, criterion, optimizer,
            scheduler, train_dl, val_dl, test_dl, checkpoint_name,
            train_on_gpu, history=[], max_epochs_stop=max_epochs_stop,
            n_epochs=n_epochs, print_every=1)

        # ── Save training checkpoint ─────────────────────────────────
        TrainChPoint = {
            'model':        model,
            'history':      history,
            'categories':   categories,
            'class_to_idx': class_to_idx,
            'idx_to_class': idx_to_class,
        }
        torch.save(TrainChPoint, save_file_name)

        # ── Plot loss & accuracy ─────────────────────────────────────
        try:
            plt.figure(figsize=(8, 6))
            for c in ['train_loss', 'val_loss', 'test_loss']:
                plt.plot(history[c], label=c)
            plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Loss')
            plt.savefig(save_path + f'/LossPerEpoch_fold_{fold_idx}.png')
            plt.close()

            plt.figure(figsize=(8, 6))
            for c in ['train_acc', 'val_acc', 'test_acc']:
                plt.plot(100 * history[c], label=c)
            plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
            plt.savefig(save_path + f'/AccuracyPerEpoch_fold_{fold_idx}.png')
            plt.close()
        except Exception as e:
            print(f'  Warning: plot saving failed ({e}) — continuing')

        del my_transforms, optimizer, scheduler
        del train_ds, train_dl, val_ds, val_dl
        del img_names_test, labels_test
        del TrainChPoint
        torch.cuda.empty_cache()

        # ==============================================================
        #  TEST EVALUATION — all 7 paper metrics
        # ==============================================================
        all_paths  = []
        test_loss  = 0.0
        i = 0
        model.eval()

        with torch.no_grad():
            for data, targets, im_path in test_dl:
                if train_on_gpu:
                    data    = data.to('cuda', non_blocking=True)
                    targets = targets.to('cuda', non_blocking=True)

                out  = model(data)
                loss = criterion(out, targets)
                test_loss += loss.item() * data.size(0)

                # Convert log-softmax → probabilities
                probs = torch.exp(out)

                all_paths.extend(im_path)
                targets_cpu = targets.cpu().numpy()
                probs_cpu   = probs.cpu().detach().numpy()
                _, preds    = torch.max(probs.cpu(), dim=1)
                preds_cpu   = preds.detach().numpy()

                if i == 0:
                    all_targets = targets_cpu
                    pred_probs  = probs_cpu
                    pred_label  = preds_cpu
                else:
                    all_targets = np.concatenate((all_targets, targets_cpu))
                    pred_probs  = np.concatenate((pred_probs,  probs_cpu))
                    pred_label  = np.concatenate((pred_label,  preds_cpu))
                i += 1

        test_loss = round(test_loss / len(test_dl.dataset), 4)

        # ── Compute & display all 7 metrics ─────────────────────────
        metrics, per_class = compute_all_metrics(
            all_targets, pred_label, pred_probs, categories)
        print_metrics(metrics, per_class, categories, fold_idx=fold_idx)
        all_fold_metrics.append(metrics)
        all_fold_per_class.append(per_class)   # ← collect per-class per fold

        # ── Save test results ────────────────────────────────────────
        from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
        cm           = confusion_matrix(all_targets, pred_label)
        cm_per_class = multilabel_confusion_matrix(all_targets, pred_label)

        save_file_name = save_path + '/' + model_name + f'_test_fold_{fold_idx}.pt'
        TestChPoint = {
            'categories':       categories,
            'class_to_idx':     class_to_idx,
            'idx_to_class':     idx_to_class,
            'Train_history':    history,
            'n_Class_test':     n_Class_test,
            'targets':          all_targets,
            'prediction_label': pred_label,
            'prediction_probs': pred_probs,
            'image_names':      all_paths,
            'cm':               cm,
            'cm_per_class':     cm_per_class,
            'metrics':          metrics,       # ← all 7 paper metrics saved
            'per_class':        per_class,     # ← per-class breakdown saved
        }
        torch.save(TestChPoint, save_file_name)

        del model, criterion, history, test_ds, test_dl
        del data, targets, out, probs
        del test_loss, loss
        del pred_probs, pred_label, all_targets, all_paths
        del cm, cm_per_class, TestChPoint
        torch.cuda.empty_cache()
        print(f'Completed fold {fold_idx}')

    print('#############################################################')

    # Delete checkpoint file
    if path.exists(checkpoint_name):
        os.remove(checkpoint_name)
        print("Checkpoint file removed!")

    # ==================================================================
    #  OVERALL SUMMARY — paper table format
    # ==================================================================

    # Re-load all test checkpoints and recompute cumulative confusion matrix
    load_path = save_path
    for fold_idx in range(loop_start, loop_end):
        fold_path   = load_path + '/' + model_name + f'_test_fold_{fold_idx}.pt'
        TestChPoint = torch.load(fold_path, weights_only=False)
        if fold_idx == loop_start:
            cumulative_cm = TestChPoint['cm']
        else:
            cumulative_cm += TestChPoint['cm']

    Overall_Accuracy = np.sum(np.diagonal(cumulative_cm)) / np.sum(cumulative_cm)
    print('\nCumulative Confusion Matrix (all folds):')
    print(cumulative_cm)
    print(f'\nOverall Test Accuracy (cumulative CM): {round(Overall_Accuracy*100, 2)}%')

    # Cross-validation summary — mean ± std across folds
    summarize_folds(all_fold_metrics, all_fold_per_class, locked_categories)

    print('#############################################################')