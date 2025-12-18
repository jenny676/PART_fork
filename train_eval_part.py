from __future__ import print_function
import os
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models.resnet import ResNet18
from models.wideresnet import WideResNet

from dataset.cifar10 import CIFAR10
from dataset.svhn import SVHN

from utils import *


parser = argparse.ArgumentParser(description='PyTorch Pixel-reweighted Adversarial Training')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument(
    '--train-percent',
    type=float,
    default=100.0,
    help='Percent of training data to use (0 < p <= 100)'
)
parser.add_argument('--epochs', type=int, default=80, metavar='N',
                    help='number of epochs to train (total, including warm-up)')
parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--epsilon', default=8/255,
                    help='maximum allowed perturbation', type=parse_fraction)
parser.add_argument('--low-epsilon', default=7/255,
                    help='maximum allowed perturbation for unimportant pixels',
                    type=parse_fraction)
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps', type=int)
parser.add_argument('--num-class', default=10,
                    help='number of classes')
parser.add_argument('--step-size', default=2/255,
                    help='perturb step size', type=parse_fraction)
parser.add_argument('--adjust-first', type=int, default=60,
                    help='adjust learning rate on which epoch in the first round')
parser.add_argument('--adjust-second', type=int, default=90,
                    help='adjust learning rate on which epoch in the second round')
parser.add_argument('--rand_init', type=bool, default=True,
                    help="whether to initialize adversarial sample with random noise")
parser.add_argument('--pre-trained', type=bool, default=False,
                    help="whether to use pre-trained weighted matrix")

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./checkpoint/ResNet_18/PART',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', default=10, type=int, metavar='N',
                    help='save frequency (still saves latest every epoch)')
parser.add_argument('--save-weights', default=1, type=int, metavar='N',
                    help='save frequency for weighted matrix')

parser.add_argument('--data', type=str, default='CIFAR10',
                    help='data source', choices=['CIFAR10', 'SVHN', 'TinyImagenet'])
parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'wideresnet'])
parser.add_argument('--warm-up', type=int, default=20, help='warm up epochs')
parser.add_argument('--cam', type=str, default='gradcam', choices=['gradcam', 'xgradcam', 'layercam'])
parser.add_argument('--attack', type=str, default='pgd', choices=['pgd', 'mma'])
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume from latest checkpoint if available')

args = parser.parse_args()


def save_checkpoint(path, model, optimizer, epoch):
    """Save model+optimizer+rng to path"""
    payload = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'rng_state': torch.get_rng_state()
    }
    if torch.cuda.is_available():
        try:
            payload['cuda_rng_state'] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    torch.save(payload, path)


def try_resume(model, optimizer, model_dir, device):
    """If --resume and latest exists, load it and return start_epoch and weighted_eps_list if present"""
    latest_path = os.path.join(model_dir, 'latest.pth')
    start_epoch = 1
    weighted_eps_list = None

    if args.resume and os.path.exists(latest_path):
        print("Resuming from checkpoint:", latest_path)
        ckpt = torch.load(latest_path, map_location=device)
        # load model & optimizer
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        # restore RNGs if present
        if 'rng_state' in ckpt:
            torch.set_rng_state(ckpt['rng_state'])
        if 'cuda_rng_state' in ckpt and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(ckpt['cuda_rng_state'])
            except Exception:
                pass
        start_epoch = ckpt.get('epoch', 1) + 1
        # try to load weighted_eps_list if exists
        wpath = os.path.join(model_dir, 'weighted_eps_latest.npy')
        if os.path.exists(wpath):
            try:
                weighted_eps_list = np.load(wpath, allow_pickle=True)
                print("Loaded weighted_eps_list from", wpath)
            except Exception as e:
                print("Could not load weighted_eps_list:", e)
    return start_epoch, weighted_eps_list


def train(args, model, device, train_loader, optimizer, epoch, weighted_eps_list):
    if args.pre_trained and isinstance(weighted_eps_list, str):
        weighted_eps_list = np.load(weighted_eps_list, allow_pickle=True)

    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        # ensure tensors on right device and correct dtype for labels
        data = data.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True).long()

        if args.pre_trained:
            # weighted_eps_list expected to be an array-like indexed by batch_idx OR precomputed per-sample
            weighted_eps = torch.from_numpy(weighted_eps_list[f'arr_{batch_idx}']).to(device)
        else:
            # if weighted_eps_list is a python list/np.array of same length as loader
            weighted_eps = weighted_eps_list[batch_idx] if weighted_eps_list is not None else None

        # calculate robust perturbation
        model.eval()
        if args.attack == 'pgd':
            data_adv = part_pgd(model,
                                data,
                                label,
                                weighted_eps,
                                epsilon=args.epsilon,
                                num_steps=args.num_steps,
                                step_size=args.step_size)
        elif args.attack == 'mma':
            data_adv = part_mma(model,
                                data,
                                label,
                                weighted_eps,
                                epsilon=args.epsilon,
                                step_size=args.step_size,
                                num_steps=args.num_steps,
                                rand_init=args.rand_init,
                                k=3,
                                num_classes=args.num_class)
        else:
            raise ValueError("Unknown attack")

        model.train()
        optimizer.zero_grad()
        out = model(data_adv)
        loss = F.cross_entropy(out, label)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            processed = (batch_idx + 1) * data.size(0)
            total = len(train_loader.dataset)
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                epoch, processed, total, 100. * (batch_idx + 1) / len(train_loader), loss.item()))


def main():
    # settings
    setup_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # optionally set CUDA devices (if you want to override)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    # setup data loader
    if args.data == 'CIFAR10':
        train_loader = CIFAR10(train_batch_size=args.batch_size).train_data()
        test_loader = CIFAR10(test_batch_size=args.batch_size).test_data()
        if args.model == 'resnet':
            model_dir = './checkpoint/CIFAR10/ResNet_18/PART'
            model = ResNet18(num_classes=10)
        elif args.model == 'wideresnet':
            model_dir = './checkpoint/CIFAR10/WideResnet-34/PART'
            model = WideResNet(34, 10, 10)
        else:
            raise ValueError("Unknown model")
    elif args.data == 'SVHN':
        args.step_size = 1/255
        args.weight_decay = 0.0035
        args.lr = 0.01
        args.batch_size = 128
        train_loader = SVHN(train_batch_size=args.batch_size).train_data()
        test_loader = SVHN(test_batch_size=args.batch_size).test_data()
        if args.model == 'resnet':
            model_dir = './checkpoint/SVHN/ResNet_18/PART'
            model = ResNet18(num_classes=10)
        elif args.model == 'wideresnet':
            model_dir = './checkpoint/SVHN/WideResnet-34/PART'
            model = WideResNet(34, 10, 10)
        else:
            raise ValueError("Unknown model")
    else:
        raise ValueError("Unknown data")

    # apply train-percent subset if requested (do this before DataParallel / samplers)
    if args.train_percent < 100.0:
        assert 0.0 < args.train_percent <= 100.0, "train-percent must be in (0,100]"
        train_dataset = train_loader.dataset
        num_train = len(train_dataset)
        num_use = max(1, int(num_train * (args.train_percent / 100.0)))
        g = torch.Generator()
        g.manual_seed(args.seed)
        perm = torch.randperm(num_train, generator=g)[:num_use].tolist()
        from torch.utils.data import Subset
        train_subset = Subset(train_dataset, perm)
        # recreate loader with common params
        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=getattr(train_loader, 'batch_size', args.batch_size),
            shuffle=True,
            num_workers=getattr(train_loader, 'num_workers', 4),
            pin_memory=getattr(train_loader, 'pin_memory', True),
            drop_last=getattr(train_loader, 'drop_last', False)
        )
        print(f"Using {num_use}/{num_train} training samples ({args.train_percent}%) for training")

    # create model, optimizer, wrap DataParallel
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # ensure model dir exists (use user-provided arg if set)
    model_dir = args.model_dir if args.model_dir else model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # try resume
    weighted_eps_list = np.load(path, allow_pickle=True)
    weighted_eps_list = [
        torch.from_numpy(w).to(device) for w in weighted_eps_list
    ]

    # warm up phase
    warmup_start = start_epoch if start_epoch <= args.warm_up else args.warm_up + 1
    if warmup_start <= args.warm_up:
        print('warm up starts')
        for epoch in range(warmup_start, args.warm_up + 1):
            standard_train(args, model, device, train_loader, optimizer, epoch)
            # always save latest
            save_checkpoint(os.path.join(model_dir, 'latest.pth'), model, optimizer, epoch)
            torch.save(model.state_dict(), os.path.join(model_dir, f'pre_part_epoch{epoch}.pth'))
            print('saved warm-up model epoch', epoch)
        print('warm up ends')
    else:
        print("Skipping warm-up (already completed in resumed checkpoint)")

    # compute or reload weighted_eps_list
    if weighted_eps_list is None:
        weighted_eps_list = save_cam(model, train_loader, device, args)
        # save latest weighted eps list
        def to_cpu_numpy(x):
          if torch.is_tensor(x):
            return x.detach().cpu().numpy()
          return x

        weighted_eps_list_cpu = [to_cpu_numpy(w) for w in weighted_eps_list]

        np.save(
            os.path.join(model_dir, 'weighted_eps_latest.npy'),
            np.array(weighted_eps_list_cpu, dtype=object),
            allow_pickle=True
        )

    else:
        print("Using weighted_eps_list loaded from checkpoint")

    # main training phase
    total_main_epochs = args.epochs - args.warm_up
    if total_main_epochs < 0:
        total_main_epochs = 0

    # compute which main epoch to start from (1-indexed as in original code)
    if start_epoch <= args.warm_up:
        main_epoch_start = 1
    else:
        main_epoch_start = start_epoch - args.warm_up

    for main_epoch in range(main_epoch_start, total_main_epochs + 1):
        # optionally recompute weighted_eps_list periodically
        if main_epoch % args.save_weights == 0 and main_epoch != 1:
            weighted_eps_list = save_cam(model, train_loader, device, args)
            np.save(os.path.join(model_dir, f'weighted_eps_epoch{main_epoch}.npy'), weighted_eps_list)
            np.save(os.path.join(model_dir, 'weighted_eps_latest.npy'), weighted_eps_list)

        # adjust learning rate
        adjust_learning_rate(args, optimizer, main_epoch)

        # adversarial training
        train(args, model, device, train_loader, optimizer, main_epoch, weighted_eps_list)

        # save checkpoint (per epoch) and latest
        epoch_global = args.warm_up + main_epoch
        save_checkpoint(os.path.join(model_dir, 'latest.pth'), model, optimizer, epoch_global)
        torch.save(model.state_dict(), os.path.join(model_dir, f'part_epoch{epoch_global}.pth'))
        if epoch_global % args.save_freq == 0:
            print('saved model at global epoch', epoch_global)

        print('================================================================')

    # evaluation on adversarial examples
    print('PGD=============================================================')
    eval_test(args, model, device, test_loader, mode='pgd')
    print('MMA==============================================================')
    eval_test(args, model, device, test_loader, mode='mma')
    print('AA==============================================================')
    eval_test(args, model, device, test_loader, mode='aa')


if __name__ == '__main__':
    main()









