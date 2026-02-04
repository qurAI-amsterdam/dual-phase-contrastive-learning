import torch
import time
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, precision_score, recall_score, accuracy_score
import os
from collections.abc import Iterable
from torch.amp import GradScaler
import datetime
import numpy as np
from collections import OrderedDict

# Automatic Mixed Precision
# https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html

# Automatic Mixed Precision examples
# https://pytorch.org/docs/stable/notes/amp_examples.html


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()


def train_one_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    device: torch.device, device_type: str, epoch: int, data_loader: Iterable,
                    scaler: GradScaler, loss_fn: torch.nn.Module, args=None):

    model.train(True)

    start_time = time.time()
    start_time_freq = time.time()

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 5 # prints every 5 iterations
    print_counts = 0

    accum_iter = args.accum_iter # increases effective batch size

    current_loss = 0

    for iter_step, data in enumerate(data_loader):

        if iter_step % accum_iter == 0:
            adjusted_lr = adjust_learning_rate(optimizer, iter_step / len(data_loader) + epoch, args)

        if args.clinical_data:
            inputs, clinical_data, labels = data
            clinical_data = clinical_data.to(device, non_blocking=True)
        else:
            inputs, labels = data

        # print(f"Inputs shape is {inputs.shape}")

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True) # , dtype=torch.float32)

        # amp is used for mixed precision training
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=args.use_amp):
            if args.clinical_data:
                outputs = model(inputs, clinical_data)
            else:
                outputs = model(inputs)
            assert outputs.dtype is torch.float16

            if labels.shape[1] > 1:
                mask = ~torch.isnan(labels).any(dim=1)
                loss = loss_fn(outputs[mask], labels[mask])
            else:
                loss = loss_fn(outputs, labels)
            assert loss.dtype is torch.float32, f"The dtype of loss is {loss.dtype}"

        # loss divided by the effective batch size
        loss /= accum_iter
        current_loss += loss

        if iter_step % accum_iter == 0:
            # Scales loss. Calls ``backward()`` on scaled loss to create scaled gradients.
            scaler.scale(current_loss).backward()

            # ``scaler.step()`` first unscales the gradients of the optimizer's assigned parameters.
            # If these gradients do not contain ``inf``s or ``NaN``s, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()
            optimizer.zero_grad() # set_to_none=True here can modestly improve performance
            last_loss = current_loss
            current_loss = 0

        if iter_step % print_freq == 0:
            total_time = str(datetime.timedelta(seconds=int(time.time() - start_time_freq)))
            start_time_freq = time.time()
            print_counts += 1
            print('{} [{}]  eta: {}  lr:{}'.format(header, str(print_counts).rjust(3), total_time, adjusted_lr))

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print('{} Total time epoch: {}'.format(header, total_time))
    print('Final Epoch Stats: loss={}, lr={}'.format(last_loss, adjusted_lr))  

    return last_loss.item()


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: Iterable, device:torch.device,
             device_type: str, loss_fn: torch.nn.Module, args=None):

    header = 'Test:'

    model.train(False)

    total_loss = 0
    all_preds = []
    all_labels = []

    start_time = time.time()

    for data in data_loader:

        inputs, labels = data

        inputs = inputs.to(device, non_blocking=True) #, dtype=torch.float32)
        labels = labels.to(device, non_blocking=True) #, dtype=torch.float32)

        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=args.use_amp): 
            outputs = model(inputs)
            assert outputs.dtype is torch.float16

            # ignore the missing row values and calculate the loss for the available labels
            mask = ~torch.isnan(labels).any(dim=1)
            loss = loss_fn(outputs[mask], labels[mask])
            total_loss += loss.item()
            assert loss.dtype is torch.float32, f"The dtype of loss is {loss.dtype}"

        all_preds.append(outputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss /  len(data_loader)
    
    all_preds = np.concatenate(all_preds, axis=0, dtype=np.float64)
    all_labels = np.concatenate(all_labels, axis=0, dtype=np.float64)

    all_preds = np.clip(all_preds, -1e6, 1e6)

    try:
        mse = mean_squared_error(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)
        r2_list = r2_score(all_labels, all_preds, multioutput='raw_values')

    except ValueError:
        print("Input contains NaN.")

        mask = ~np.isnan(all_labels).any(axis=1) & ~np.isnan(all_preds).any(axis=1)
        # Apply the mask
        all_labels = all_labels[mask]
        all_preds = all_preds[mask]

        print(all_labels.shape)
        print(all_preds.shape)

        mse = mean_squared_error(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        r2_list = r2_score(all_labels, all_preds, multioutput='raw_values')
        print("r2 for the different outputs:")
        print(r2_list)
        r2 = r2_score(all_labels, all_preds)

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print('{} Total time epoch: {}'.format(header, total_time))
    print('Stats: average loss={}, mse={}, mae={}, r2={}\n'.format(avg_loss, mse, mae, r2))

    # match the generated outputs with the training labels
    if int(args.num_outputs) == 76:

        r2_labels = ['LV end diastolic volume', 'LV end systolic volume', 'LV stroke volume', 'LV ejection fraction', 'LV cardiac output',
                    'LV myocardial mass', 'RV end diastolic volume', 'RV end systolic volume', 'RV stroke volume', 'RV ejection fraction',
                    'LA maximum volume', 'LA minimum volume', 'LA stroke volume', 'LA ejection fraction', 'RA maximum volume',
                    'RA minimum volume', 'RA stroke volume', 'RA ejection fraction', 'LV mean myocardial wall thickness AHA 1',
                    'LV mean myocardial wall thickness AHA 2', 'LV mean myocardial wall thickness AHA 3', 'LV mean myocardial wall thickness AHA 4', 'LV mean myocardial wall thickness AHA 5', 'LV mean myocardial wall thickness AHA 6',
                    'LV mean myocardial wall thickness AHA 7', 'LV mean myocardial wall thickness AHA 8', 'LV mean myocardial wall thickness AHA 9', 'LV mean myocardial wall thickness AHA 10', 'LV mean myocardial wall thickness AHA 11',
                    'LV mean myocardial wall thickness AHA 12', 'LV mean myocardial wall thickness AHA 13', 'LV mean myocardial wall thickness AHA 14', 'LV mean myocardial wall thickness AHA 15', 'LV mean myocardial wall thickness AHA 16',
                    'LV mean myocardial wall thickness global', 'LV circumferential strain AHA 1', 'LV circumferential strain AHA 2', 'LV circumferential strain AHA 3', 'LV circumferential strain AHA 4',
                    'LV circumferential strain AHA 5', 'LV circumferential strain AHA 6', 'LV circumferential strain AHA 7', 'LV circumferential strain AHA 8', 'LV circumferential strain AHA 9',
                    'LV circumferential strain AHA 10', 'LV circumferential strain AHA 11', 'LV circumferential strain AHA 12', 'LV circumferential strain AHA 13', 'LV circumferential strain AHA 14',
                    'LV circumferential strain AHA 15', 'LV circumferential strain AHA 16', 'LV circumferential strain global', 'LV radial strain AHA 1', 'LV radial strain AHA 2',
                    'LV radial strain AHA 3', 'LV radial strain AHA 4', 'LV radial strain AHA 5', 'LV radial strain AHA 6', 'LV radial strain AHA 7',
                    'LV radial strain AHA 8', 'LV radial strain AHA 9', 'LV radial strain AHA 10', 'LV radial strain AHA 11', 'LV radial strain AHA 12',
                    'LV radial strain AHA 13', 'LV radial strain AHA 14', 'LV radial strain AHA 15', 'LV radial strain AHA 16', 'LV radial strain global',
                    'LV longitudinal strain Segment 1', 'LV longitudinal strain Segment 2', 'LV longitudinal strain Segment 3', 'LV longitudinal strain Segment 4', 'LV longitudinal strain Segment 5',
                    'LV longitudinal strain Segment 6', 'LV longitudinal strain global']

    elif int(args.num_outputs) == 82:

        r2_labels = ['LV end diastolic volume', 'LV end systolic volume', 'LV stroke volume', 'LV ejection fraction', 'LV cardiac output',
                    'LV myocardial mass', 'RV end diastolic volume', 'RV end systolic volume', 'RV stroke volume', 'RV ejection fraction',
                    'LA maximum volume', 'LA minimum volume', 'LA stroke volume', 'LA ejection fraction', 'RA maximum volume',
                    'RA minimum volume', 'RA stroke volume', 'RA ejection fraction', 'Ascending aorta max area', 'Ascending aorta min area',
                    'Ascending aorta distensibility', 'Descending aorta distensibility', 'Descending aorta minimum area', 'Descending aorta distensibility', 'LV mean myocardial wall thickness AHA 1',
                    'LV mean myocardial wall thickness AHA 2', 'LV mean myocardial wall thickness AHA 3', 'LV mean myocardial wall thickness AHA 4', 'LV mean myocardial wall thickness AHA 5', 'LV mean myocardial wall thickness AHA 6',
                    'LV mean myocardial wall thickness AHA 7', 'LV mean myocardial wall thickness AHA 8', 'LV mean myocardial wall thickness AHA 9', 'LV mean myocardial wall thickness AHA 10', 'LV mean myocardial wall thickness AHA 11',
                    'LV mean myocardial wall thickness AHA 12', 'LV mean myocardial wall thickness AHA 13', 'LV mean myocardial wall thickness AHA 14', 'LV mean myocardial wall thickness AHA 15', 'LV mean myocardial wall thickness AHA 16',
                    'LV mean myocardial wall thickness global', 'LV circumferential strain AHA 1', 'LV circumferential strain AHA 2', 'LV circumferential strain AHA 3', 'LV circumferential strain AHA 4',
                    'LV circumferential strain AHA 5', 'LV circumferential strain AHA 6', 'LV circumferential strain AHA 7', 'LV circumferential strain AHA 8', 'LV circumferential strain AHA 9',
                    'LV circumferential strain AHA 10', 'LV circumferential strain AHA 11', 'LV circumferential strain AHA 12', 'LV circumferential strain AHA 13', 'LV circumferential strain AHA 14',
                    'LV circumferential strain AHA 15', 'LV circumferential strain AHA 16', 'LV circumferential strain global', 'LV radial strain AHA 1', 'LV radial strain AHA 2',
                    'LV radial strain AHA 3', 'LV radial strain AHA 4', 'LV radial strain AHA 5', 'LV radial strain AHA 6', 'LV radial strain AHA 7',
                    'LV radial strain AHA 8', 'LV radial strain AHA 9', 'LV radial strain AHA 10', 'LV radial strain AHA 11', 'LV radial strain AHA 12',
                    'LV radial strain AHA 13', 'LV radial strain AHA 14', 'LV radial strain AHA 15', 'LV radial strain AHA 16', 'LV radial strain global',
                    'LV longitudinal strain Segment 1', 'LV longitudinal strain Segment 2', 'LV longitudinal strain Segment 3', 'LV longitudinal strain Segment 4', 'LV longitudinal strain Segment 5',
                    'LV longitudinal strain Segment 6', 'LV longitudinal strain global']

    elif int(args.num_outputs) == 3:
        r2_labels = ["LV end diastolic volume",  "LV myocardial mass",  "RV end diastolic volume"]

    elif int(args.num_outputs) == 5:
        r2_labels = ['LV end diastolic volume', 'LV end systolic volume', 'LV ejection fraction', 'RV end diastolic volume', 'RV end systolic volume']

    elif int(args.num_outputs) == 13:
        r2_labels = ['LV end diastolic volume', 'LV end systolic volume', 'LV stroke volume', 'LV ejection fraction', 'LV cardiac output', 'LV myocardial mass', 'RV end diastolic volume',
                   'RV end systolic volume', 'RV stroke volume', 'RV ejection fraction', 'LV mean myocardial wall thickness global', 'LV circumferential strain global',
                   'LV radial strain global']
    elif int(args.num_outputs) == 1:
        r2_labels = ["output"]

    for i in range(len(r2_labels)):
        print(f'{r2_labels[i]}: {r2_list[i]}')

    metrics = {
        'avg_loss': avg_loss,
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

    return metrics


@torch.no_grad()
def evaluate_fine_tune(model: torch.nn.Module, data_loader: Iterable, device:torch.device,
             device_type: str, loss_fn: torch.nn.Module, args=None):

    header = 'Test:'

    model.train(False)

    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    start_time = time.time()

    for data in data_loader:

        if args.clinical_data:
            inputs, clinical_data, labels = data
            clinical_data = clinical_data.to(device, non_blocking=True)
        else:
            inputs, labels = data

        inputs = inputs.to(device, non_blocking=True) #, dtype=torch.float32)
        labels = labels.to(device, non_blocking=True) #, dtype=torch.float32)

        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=args.use_amp):
            if args.clinical_data:
                outputs = model(inputs, clinical_data)
            else:
                outputs = model(inputs)
            assert outputs.dtype is torch.float16

            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            assert loss.dtype is torch.float32, f"The dtype of loss is {loss.dtype}"

        probs = torch.sigmoid(outputs)
        all_preds.append(outputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss /  len(data_loader)
    
    all_preds = np.concatenate(all_preds, axis=0, dtype=np.float64)
    all_labels = np.concatenate(all_labels, axis=0, dtype=np.float64)
    all_probs = np.concatenate(all_probs, axis=0, dtype=np.float64)

    all_preds = np.clip(all_probs, -1e6, 1e6)
    all_preds = (all_preds > 0).astype(int)

    roc = roc_auc_score(all_labels, all_probs)
    
    pre = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print('{} Total time epoch: {}'.format(header, total_time))
    print('Stats: average loss={}, roc score={}, accuracy={}, precision={}, recall={}\n'.format(avg_loss, roc, acc, pre, rec))  

    metrics = {
        'avg_loss': avg_loss,
        'accuracy': acc,
        'roc': roc,
        'precision': pre,
        'recall': rec
    }

    return metrics


def load_checkpoint(model, checkpoint_path, device, optimizer=None):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Model checkpoint loaded succesfully")

        return checkpoint

    else:
        print(f"Not checkpoint provided. Starting model training with default model.")

def load_resnet50(cmr_model, checkpoint_path_cmr, device):
    try:
        load_checkpoint(cmr_model, checkpoint_path_cmr, device)
    except:
        print("Loading model with new keys to match model keys...")
        checkpoint = torch.load(checkpoint_path_cmr, map_location=device)
        print(checkpoint['model_state_dict'].keys())
        encoder_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith('model.0'):
                new_k = "model.conv1" + k[7:]
                encoder_state_dict[new_k] = v
            elif k.startswith('model.1'):
                new_k = "model.bn1" + k[7:]
                encoder_state_dict[new_k] = v
            elif k.startswith('model.4'):
                new_k = "model.layer1" + k[7:]
                encoder_state_dict[new_k] = v
            elif k.startswith('model.5'):
                new_k = "model.layer2" + k[7:]
                encoder_state_dict[new_k] = v
            elif k.startswith('model.6'):
                new_k = "model.layer3" + k[7:]
                encoder_state_dict[new_k] = v
            elif k.startswith('model.7'):
                new_k = "model.layer4" + k[7:]
                encoder_state_dict[new_k] = v
            elif k.startswith('encoder'):
                encoder_state_dict[k] = v
            elif k.startswith('output_layer'):
                encoder_state_dict[k] = v
            elif k in ['fc.weight', 'fc.bias']:
                encoder_state_dict[k] = v

        cmr_model.load_state_dict(encoder_state_dict, strict=False)
        print("Pre-trained CMR encoder loaded")


def load_resnet50_3D(cmr_model, checkpoint_path_cmr, device):
    try:
        load_checkpoint(cmr_model, checkpoint_path_cmr, device)
    except:
        checkpoint = torch.load(checkpoint_path_cmr, map_location=device)

        encoder_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith('model'):
                new_k = "model.blocks" + k[5:]
                encoder_state_dict[new_k] = v
            else:
                encoder_state_dict[k] = v
    
        cmr_model.load_state_dict(encoder_state_dict)
        print("Pre-trained CMR encoder loaded")


def save_checkpoint(model, checkpoint_path: str, optimizer=None, epoch=None,
                    loss=None, args=None):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'epoch': epoch,
        'loss': loss,
        'model_class': model.__class__.__name__,  # Store model class name
    }

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    for filename in os.listdir(checkpoint_path):
        if filename.endswith('.pth'):
            # eliminate other checkpoint in path
            os.remove(os.path.join(checkpoint_path, filename))

    checkpoint_name = os.path.join(checkpoint_path,
        f"{model.__class__.__name__}_checkpoint-{epoch}-loss-{loss}-lr-{args.lr}-wd-{args.weight_decay}-ai-{args.accum_iter}.pth"
    )
    torch.save(checkpoint, checkpoint_name)

    print(f"{model.__class__.__name__} model's checkpoint saved at {checkpoint_path}")