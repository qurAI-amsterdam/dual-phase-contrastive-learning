import torch
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple
import argparse
from engine_cl import train_one_epoch, evaluate, get_rank, get_world_size, load_checkpoint, save_checkpoint
from dataset_cl import CLDataset
from models_cl import get_model, ClipContrastiveLearning, MultiModalClipContrastiveLearning
from utils.clip_loss import CLIPLoss, MultiModalCLIPLoss


def get_args_parser():
    parser = argparse.ArgumentParser('CMR pre-training', add_help=False)

    # Main arguments
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--model_name', default="ResNet50", choices=["SwinTransformer", "ResNet50", "ResNet50-3D", "ResNet50-4D", "ResNet50-3D-MLP"])
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # CMR model argument
    parser.add_argument('--ecg_input_size', default=(12,5000)) # ECG input size (MRI) (100,300)
    parser.add_argument('--num_outputs', default=1)
    parser.add_argument('--pretrained', default=True, help='If the CMR model uses pretrained weight. By default, it uses them.')
    parser.add_argument('--temporal_dim', type=int, default=50)
    parser.add_argument('--hidden_proj_dim_cmr', type=int, default=768)


    # ECG model argument
    parser.add_argument('--ecg_model', default='vit_base_patch200', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--patch_height', default=1, type=int, metavar='N',
                        help='patch height')
    parser.add_argument('--patch_width', default=100, type=int, metavar='N',
                        help='patch width')
    parser.add_argument('--patch_size', default=(1, 100), type=Tuple,
                        help='patch size')
    parser.add_argument('--drop_path', default=0.1)
    parser.add_argument('--global_pool', default=True, type=bool)
    parser.add_argument('--input_size', default=1, type=int) # number of channels in the ECG
    parser.add_argument('--hidden_proj_dim_ecg', type=int, default=432)

    # Data & path arguments
    parser.add_argument('--cmr_train_path', type=str, help='Path to the train dataset for the CMR data')
    parser.add_argument('--cmr_systole_train_path', type=str, default=None, help='Path to the train dataset for the CMR data')
    parser.add_argument('--ecg_train_path', type=str, help='Path to the train dataset for the ECG data')
    parser.add_argument('--cmr_val_path', type=str, help='Path to the validation dataset for the CMR data')
    parser.add_argument('--cmr_systole_val_path', type=str, default=None, help='Path to the val dataset for the CMR data')
    parser.add_argument('--ecg_val_path', type=str, help='Path to the validation dataset for the ECG data')
    parser.add_argument('--output_dir', default=None,
                        help='Path to save model checkpoints and results. If None, the checkpoints and results are not saved.')
    parser.add_argument('--checkpoint_path', default='',
                        help='Path to the saved model checkpoint. If None, training from scratch.')
    parser.add_argument('--checkpoint_path_cmr', default='',
                        help='Path to the pre-trained CMR encoder checkpoint. If None, training from scratch.')
    parser.add_argument('--checkpoint_path_cmr_systole', default=None,
                        help='Path to the pre-trained CMR encoder checkpoint when doing MultiModal contrastive learning.')
    parser.add_argument('--checkpoint_path_ecg', default='',
                        help='Path to the pre-trained ECG encoder checkpoint. If None, training from scratch.')

    # Optimizer arguments
    parser.add_argument('--weight_decay', default=0.005)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--early_stopping', default=True, type=bool)
    parser.add_argument('--patience', default=15, type=int)
    parser.add_argument('--warmup_epochs', default=40, type=int)
    parser.add_argument('--min_lr', type=float, default=0.)

    # Efficiency arguments
    parser.add_argument('--use_amp', default=True)
    parser.add_argument('--pin_memory', default=True, action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--num_workers', default=18, type=int)

    # Contrastive Learning arguments
    parser.add_argument('--return_latent_space', default=True)
    parser.add_argument('--contrastive_learning', default=True)
    parser.add_argument('--latent_dim', default=768)
    parser.add_argument('--lambda_', default=0.5, type=float,
                        help='Balances the loss between the ECG and the CMR.')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='Controls the range of the logits in the softmax function.')
    parser.add_argument('--projection_dim', default=128, type=int)
    parser.add_argument('--encode_temporal_dimension', default=True,
                        help="Rather to apply a transformer encoder to the temporal dimensions or not")
    
    # Triple CL Arguments
    parser.add_argument("--lambda_ecg_cmr1", default=0.33, help="Lambda Value between the ECG and the CMR.")
    parser.add_argument("--lambda_ecg_cmr2", default=0.33, help="Lambda Value between the ECG and the CMR.")
    parser.add_argument("--lambda_cmr1_cmr2", default=0.33, help="Lambda Value between the two CMR inputs.")

    # Fine-tune arguments
    parser.add_argument('--train_labels_path', type=str, help='Path to the train dataset', default=None)
    parser.add_argument('--val_labels_path', type=str, help='Path to the validation dataset', default=None)
    parser.add_argument('--test_labels_path', type=str, help='Path to the test dataset', default=None)

    # Reproducibility
    parser.add_argument('--seed', default=42)

    return parser


def main(args):
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Training on {device_type}")
    # torch.set_default_device(device)

    device = torch.device(device_type)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # train augmentations
    cmr_transform_train = transforms.Compose([
            # transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            #transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])]) # grey-scale images

    cmr_transform_val = transforms.Compose([
            # transforms.Resize(args.input_size, interpolation=3),  # 3 is bicubic
            #transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])]) # grey-scale images

    # TODO change the dataset for CLDataset, once CLDataset is defined
    dataset_train = CLDataset(cmr_data_path=args.cmr_train_path, ecg_data_path=args.ecg_train_path,
                              cmr_data_path_systole=args.cmr_systole_train_path, 
                              labels_path=args.train_labels_path, train=True, 
                              cmr_transform=cmr_transform_train, ecg_transform=None)

    dataset_val = CLDataset(cmr_data_path=args.cmr_val_path, ecg_data_path=args.ecg_val_path,
                            cmr_data_path_systole=args.cmr_systole_val_path,
                            labels_path=args.val_labels_path, train=True, 
                            cmr_transform=cmr_transform_train, ecg_transform=None)

    num_tasks = get_world_size()
    global_rank = get_rank()

    # g = torch.Generator(device=device)
    
    # sampler_train = torch.utils.data.DistributedSampler(
    #     dataset_train, num_replicas=num_tasks,
    #     rank=global_rank, shuffle=True, # drop_last=False
    # )

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks,
        rank=global_rank, shuffle=True, # drop_last=False
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=None,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    # get model retrieves the CMR model (SwinTransformer or RestNet50)
    print("args.input_size", args.input_size)
    args.len_data_loader = len(data_loader_train.dataset) 
    print("Train dataloader size:", args.len_data_loader)

    if args.checkpoint_path_cmr_systole is not None:
        model = MultiModalClipContrastiveLearning(args.input_size, args.return_latent_space,
                 args.pretrained, args.contrastive_learning,
                 args.latent_dim, args=args)
        loss_fn = MultiModalCLIPLoss(temperature=args.temperature, lambda_ecg_cmr1=args.lambda_ecg_cmr1, 
             lambda_ecg_cmr2=args.lambda_ecg_cmr2, lambda_cmr1_cmr2=args.lambda_cmr1_cmr2)

    else:
        model = ClipContrastiveLearning(args.input_size, args.return_latent_space,
                 args.pretrained, args.contrastive_learning,
                 args.latent_dim, args=args)
        loss_fn = CLIPLoss(temperature=args.temperature, lambda_0=args.lambda_)

    # get_model(model_name=args.model_name, args=args)

    model.to(device, non_blocking=True)

    eff_batch_size = args.batch_size * args.accum_iter * get_world_size()
    print(f"Effective batch size given batch size and accumulate gradient iterations is {eff_batch_size}.")

    # param_groups = optim_factory.add_weight_decay(model, args.weight_decay)    
    # optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print("args.weight_decay", float(args.weight_decay))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=float(args.weight_decay))

    grad_scaler = torch.amp.GradScaler(enabled=args.use_amp)

    # Variables to track training progress and checkpoint loading
    start_epoch = 0
    best_loss = float("inf")
    epochs_wo_improvement = 0

    loss_fn_classification = torch.nn.BCEWithLogitsLoss() # classification

    # load CL models from checkpoints if path is provided
    if args.checkpoint_path:
        checkpoint_data = load_checkpoint(model, args.checkpoint_path, device, optimizer)
        if checkpoint_data and 'epoch' in checkpoint_data: # retrieve epoch from checkpoint
            start_epoch = checkpoint_data['epoch'] + 1
        if checkpoint_data and 'loss' in checkpoint_data:  # retrieve loss from checkpoint
            best_loss = checkpoint_data['loss']

    for epoch in range(start_epoch, args.epochs):
        # data_loader_train.sampler.set_epoch(epoch) # TODO check if makes training to start from last epoch when loading a model

        sampler_train.set_epoch(epoch)

        train_metrics = train_one_epoch(clip_cl=model, optimizer=optimizer,
            device=device, device_type=device_type, epoch=epoch, data_loader=data_loader_train,
            scaler=grad_scaler, loss_fn=loss_fn, loss_fn_classification=loss_fn_classification, args=args)

        eval_metrics = evaluate(clip_cl=model, data_loader=data_loader_val, device=device,
                                device_type=device_type, loss_fn=loss_fn,
                                loss_fn_classification=loss_fn_classification, args=args)

        # if args.eval_criterion == "loss":
        if eval_metrics['avg_loss'] < best_loss:
            best_loss = eval_metrics['avg_loss']
            epochs_wo_improvement = 0

            # save the model with best loss if output directed is provided
            if args.output_dir:
                save_checkpoint(model, args.output_dir, optimizer, epoch, loss=best_loss)
                print(f"Saved best model at epoch {epoch} with loss {best_loss:.4f}")
        else:
            epochs_wo_improvement +=1
            if epochs_wo_improvement > args.patience and args.early_stopping:
                print(f"Early stopping triggered after {epochs_wo_improvement} epochs without improvement")
                break


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)
