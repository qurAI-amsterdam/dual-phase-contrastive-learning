import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import argparse
from engine_cmr import train_one_epoch, evaluate, get_rank, get_world_size, load_checkpoint, save_checkpoint, load_resnet50, load_resnet50_3D
from cmr_dataset import CMRDataset
from models_cmr import get_model


def get_args_parser():
    parser = argparse.ArgumentParser('CMR pre-training', add_help=False)

    # Main arguments
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--model_name', default="ResNet50", choices=["SwinTransformer", "ResNet50", "ResNet50-3D", "ResNet50-3D-MLP"])
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--input_size', default=1) # grayscale images (MRI) (100,300)
    parser.add_argument('--num_outputs', default=1)
    parser.add_argument('--pretrained', default=True, help='If the model uses pretrained weight. By default, it uses them.')
    parser.add_argument('--min_lr', type=float, default=0.)
    parser.add_argument('--temporal_dim', type=int, default=50)

    # CMR model argument

    # Data & path arguments
    parser.add_argument('--train_path', type=str, help='Path to the train dataset')
    parser.add_argument('--val_path', type=str, help='Path to the validation dataset')
    parser.add_argument('--test_path', type=str, help='Path to the test dataset')
    parser.add_argument('--train_labels_path', type=str, help='Path to the labels of the train dataset')
    parser.add_argument('--val_labels_path', type=str, help='Path to the labels of the validation dataset')
    parser.add_argument('--test_labels_path', type=str, help='Path to the labels of the test dataset')
    parser.add_argument('--output_dir', default=None,
                        help='Path to save model checkpoints and results. If None, the checkpoints and results are not saved.')
    parser.add_argument('--checkpoint_path', default='',
                        help='Path to the saved model checkpoint. If None, training from scratch.')
    parser.add_argument('--clinical_data', default=False)

    # Optimizer arguments
    parser.add_argument('--weight_decay', default=0.005)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--early_stopping', default=True, type=bool)
    parser.add_argument('--patience', default=15, type=int)
    parser.add_argument('--warmup_epochs', default=40, type=int)

    # Efficiency arguments
    parser.add_argument('--use_amp', default=True)
    parser.add_argument('--pin_memory', default=True, action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--num_workers', default=15, type=int)

    # Contrastive Learning arguments
    parser.add_argument('--return_latent_space', default=False)
    parser.add_argument('--contrastive_learning', default=False)
    parser.add_argument('--latent_dim', default=768)

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
    transform_train = transforms.Compose([
            # transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            #transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])]) # grey-scale images

    transform_val = transforms.Compose([
            # transforms.Resize(args.input_size, interpolation=3),  # 3 is bicubic
            #transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])]) # grey-scale images

    dataset_train = CMRDataset(data_path=args.train_path, labels_path=args.train_labels_path,
                 train=True, transform=transform_train)

    print("len dataset_train:", len(dataset_train))

    dataset_val = CMRDataset(data_path=args.val_path, labels_path=args.val_labels_path,
                 train=False, transform=transform_val)
    
    print("len dataset_val:", len(dataset_val))

    dataset_test = CMRDataset(data_path=args.test_path, labels_path=args.test_labels_path,
                 train=False, transform=transform_val)

    print("len dataset_test:", len(dataset_test))

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

    # Create generator with correct device
    if torch.cuda.is_available():
        generator = torch.Generator(device='cuda')
    else:
        generator = torch.Generator(device='cpu')

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        # generator=generator,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=None,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        # generator=generator,
        drop_last=False,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=None,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        # generator=generator,
        drop_last=False,
    )

    # get model retrieves the CMR model (SwinTransformer or RestNet50)
    model = get_model(model_name=args.model_name, args=args)

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

    loss_fn = torch.nn.MSELoss()

    # load models from checkpoints if path is provided
    if args.checkpoint_path:
        checkpoint_data = load_checkpoint(model, args.checkpoint_path, device, optimizer)
        if checkpoint_data and 'epoch' in checkpoint_data: # retrieve epoch from checkpoint
            start_epoch = checkpoint_data['epoch'] + 1
        if checkpoint_data and 'loss' in checkpoint_data:  # retrieve loss from checkpoint
            best_loss = checkpoint_data['loss']

    eval_metrics_dic = {}
    train_loss_list = []
    eval_metrics_df = pd.DataFrame()
    for epoch in range(start_epoch, args.epochs):

        sampler_train.set_epoch(epoch)

        train_metrics = train_one_epoch(model=model, optimizer=optimizer,
            device=device, device_type=device_type, epoch=epoch, data_loader=data_loader_train,
            scaler=grad_scaler, loss_fn=loss_fn, args=args)

        eval_metrics = evaluate(model=model, data_loader=data_loader_val, device=device,
                                device_type=device_type, loss_fn=loss_fn, args=args)
        eval_metrics_dic[epoch] = eval_metrics
        eval_metrics_epoch = pd.DataFrame([eval_metrics])
        eval_metrics_df = pd.concat([eval_metrics_df, eval_metrics_epoch], ignore_index=True)
        train_loss_list.append(train_metrics)

        if eval_metrics['avg_loss'] < best_loss:
            best_loss = eval_metrics['avg_loss']
            epochs_wo_improvement = 0

            # save the model with best loss if output directed is provided
            if args.output_dir:
                save_checkpoint(model, args.output_dir, optimizer, epoch, loss=best_loss, args=args)
                print(f"Saved best model at epoch {epoch} with loss {best_loss:.4f}\n\n")
        else:
            epochs_wo_improvement +=1
            if epochs_wo_improvement > args.patience and args.early_stopping:
                print(f"Early stopping triggered after {epochs_wo_improvement} epochs without improvement")
                break

    print(eval_metrics_dic)
    print(train_loss_list)

    eval_metrics_df.to_csv(f"evaluation_metrics_{args.model_name}.csv")

    print("\nFinal evaluation metrics on test set:")
    test_metrics = evaluate(model=model, data_loader=data_loader_test, device=device,
                        device_type=device_type, loss_fn=loss_fn, args=args)


def evualte_test_set(args):
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Training on {device_type}")
    # torch.set_default_device(device)

    device = torch.device(device_type)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    transform_test = transforms.Compose([
            # transforms.Resize(args.input_size, interpolation=3),  # 3 is bicubic
            #transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])]) # grey-scale images
    
    dataset_test = CMRDataset(data_path=args.test_path, labels_path=args.test_labels_path,
                 train=False, transform=transform_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=None,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        # generator=generator,
        drop_last=False,
    )

    # get model retrieves the CMR model (SwinTransformer or RestNet50)
    model = get_model(model_name=args.model_name, args=args)

    model.to(device, non_blocking=True)

    eff_batch_size = args.batch_size * args.accum_iter * get_world_size()
    print(f"Effective batch size given batch size and accumulate gradient iterations is {eff_batch_size}.")

    # param_groups = optim_factory.add_weight_decay(model, args.weight_decay)    
    # optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print("args.weight_decay", float(args.weight_decay))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=float(args.weight_decay))

    grad_scaler = torch.amp.GradScaler(enabled=args.use_amp)

    loss_fn = torch.nn.MSELoss()

    # load models from checkpoints if path is provided
    if args.checkpoint_path:
        load_checkpoint(model, args.checkpoint_path, device, optimizer)
    else:
        print("No module loaded and evaluating a non-trained model.")

    print("\nFinal evaluation metrics on test set:")
    test_metrics = evaluate(model=model, data_loader=data_loader_test, device=device,
                        device_type=device_type, loss_fn=loss_fn, args=args)

    return test_metrics['r2']

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    # main(args)

    evualte_test_set(args)
