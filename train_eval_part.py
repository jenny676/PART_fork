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

import tempfile
import shutil

def safe_numpy_save(path, arr, allow_pickle=True):
    """Atomically save arr to path using a temp file and os.replace()."""
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=d, prefix='.tmp_save_', suffix='.npy')
    os.close(fd)
    try:
        with open(tmp_path, 'wb') as f:
            np.save(f, arr, allow_pickle=allow_pickle)
        os.replace(tmp_path, path)  # atomic on POSIX
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise

def safe_load_weighted_eps(weighted_eps_path, model, train_loader, device, args):
    """Try to load weighted_eps; if missing/corrupt or ill-sized, recompute and save atomically.

    Ensures the returned list/sequence has length == len(train_loader) (one entry per batch).
    Converts stored numpy/tensor elements to torch tensors on `device`.
    """
    def recompute_and_save():
        print("Recomputing weighted_eps_list via save_cam(...)")
        w = save_cam(model, train_loader, device, args)
        # convert tensors to CPU numpy arrays for storage (object array)
        w_cpu = []
        for t in w:
            if torch.is_tensor(t):
                w_cpu.append(t.detach().cpu().numpy())
            else:
                w_cpu.append(np.array(t, dtype=object))
        safe_numpy_save(weighted_eps_path, np.array(w_cpu, dtype=object))
        # return as list of torch tensors on device (per-batch)
        ret = []
        for arr in w_cpu:
            try:
                ret.append(torch.from_numpy(np.asarray(arr)).to(device))
            except Exception:
                # fallback: keep as numpy array converted to tensor
                ret.append(torch.as_tensor(np.array(arr, dtype=object)).to(device))
        return ret

    # If file missing or empty -> recompute
    if not os.path.exists(weighted_eps_path):
        print("weighted_eps file not found -> recomputing.")
        return recompute_and_save()

    if os.path.getsize(weighted_eps_path) == 0:
        print("weighted_eps file is empty -> recomputing.")
        try:
            os.remove(weighted_eps_path)
        except Exception:
            pass
        return recompute_and_save()

    try:
        arr = np.load(weighted_eps_path, allow_pickle=True)
        if arr is None or len(arr) == 0:
            print("Loaded arr empty -> recomputing.")
            return recompute_and_save()

        # If the stored arr length doesn't match number of batches, recompute.
        expected_batches = len(train_loader)
        if len(arr) != expected_batches:
            print(f"weighted_eps length mismatch (loaded {len(arr)} != expected {expected_batches}) -> recomputing.")
            return recompute_and_save()

        # convert to torch tensors on device
        result = []
        for item in arr:
            # item may be ndarray, list, or already tensor
            if isinstance(item, np.ndarray):
                try:
                    result.append(torch.from_numpy(item).to(device))
                except Exception:
                    # fallback: convert object -> array then tensor
                    result.append(torch.as_tensor(np.array(item, dtype=object)).to(device))
            elif torch.is_tensor(item):
                result.append(item.to(device))
            else:
                # try to coerce arbitrary object to numpy then tensor
                try:
                    tmp = np.array(item)
                    result.append(torch.from_numpy(tmp).to(device))
                except Exception:
                    print("Element conversion failed -> recomputing.")
                    return recompute_and_save()

        print("Loaded weighted_eps_list from", weighted_eps_path)
        return result

    except (EOFError, ValueError, Exception) as e:
        print("Failed to load weighted_eps_list:", repr(e))
        return recompute_and_save()


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
parser.add_argument('--eval-modes', type=str, default='all',
                    help="Comma-separated eval modes to run: 'pgd', 'mma', 'aa', or 'all' (default)")
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
    import numpy as _np  # local import to be safe if np isn't imported at top
    latest_path = os.path.join(model_dir, 'latest.pth')
    start_epoch = 1
    weighted_eps_list = None

    if args.resume and os.path.exists(latest_path):
        print("Resuming from checkpoint:", latest_path)
        ckpt = torch.load(latest_path, map_location=device)
        # load model & optimizer
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])

        # ---------- restore RNGs if present (robust) ----------
        def _to_byte_tensor(x):
            """Convert x (bytes, bytearray, list, ndarray, or torch tensor) -> CPU torch.uint8 tensor."""
            if isinstance(x, torch.Tensor):
                t = x.to(dtype=torch.uint8).cpu()
                return t
            if isinstance(x, (bytes, bytearray)):
                # bytes -> list of ints -> tensor
                return torch.as_tensor(list(x), dtype=torch.uint8).cpu()
            if isinstance(x, _np.ndarray):
                arr = x.astype(_np.uint8, copy=False)
                return torch.from_numpy(arr).to(dtype=torch.uint8).cpu()
            if isinstance(x, (list, tuple)):
                # list of ints or nested structure: try flat conversion
                try:
                    return torch.as_tensor(x, dtype=torch.uint8).cpu()
                except Exception:
                    # last resort: convert via numpy
                    return torch.from_numpy(_np.array(x, dtype=_np.uint8)).cpu()
            # unknown type
            raise TypeError(f"Unsupported RNG state type: {type(x)}")

        # safe restore for CPU rng_state
        if 'rng_state' in ckpt:
            try:
                rng_state = ckpt['rng_state']
                # convert to byte tensor if needed
                if not isinstance(rng_state, torch.Tensor):
                    rng_state = _to_byte_tensor(rng_state)
                else:
                    rng_state = rng_state.to(dtype=torch.uint8).cpu()
                torch.set_rng_state(rng_state)
            except Exception as e:
                print(f"Warning: could not restore CPU RNG state ({e}), continuing without it.")

        # safe restore for CUDA RNGs (if available)
        if 'cuda_rng_state' in ckpt and torch.cuda.is_available():
            try:
                cuda_state = ckpt['cuda_rng_state']
                # if it's a list/tuple of states for each device, convert each element
                if isinstance(cuda_state, (list, tuple)):
                    converted = []
                    for s in cuda_state:
                        if not isinstance(s, torch.Tensor):
                            converted.append(_to_byte_tensor(s))
                        else:
                            converted.append(s.to(dtype=torch.uint8).cpu())
                    torch.cuda.set_rng_state_all(converted)
                else:
                    # single-state case: convert and set for all devices
                    s = cuda_state
                    if not isinstance(s, torch.Tensor):
                        s = _to_byte_tensor(s)
                    else:
                        s = s.to(dtype=torch.uint8).cpu()
                    # attempt to broadcast same state to all devices
                    try:
                        torch.cuda.set_rng_state_all([s])
                    except Exception:
                        # fallback: try set for current device only
                        try:
                            torch.cuda.set_rng_state(s)
                        except Exception as e:
                            raise e
            except Exception as e:
                print(f"Warning: could not restore CUDA RNG state ({e}), continuing without it.")

        # ------------------------------------------------------

        start_epoch = ckpt.get('epoch', 1) + 1

        # try to load weighted_eps_list if exists
        wpath = os.path.join(model_dir, 'weighted_eps_latest.npy')
        if os.path.exists(wpath):
            try:
                weighted_eps_list = _np.load(wpath, allow_pickle=True)
                print("Loaded weighted_eps_list from", wpath)
            except Exception as e:
                print("Could not load weighted_eps_list:", e)
    return start_epoch, weighted_eps_list


def train(args, model, device, train_loader, optimizer, epoch, weighted_eps_list):
    if args.pre_trained and isinstance(weighted_eps_list, str):
        weighted_eps_list = np.load(weighted_eps_list, allow_pickle=True)

    model.train()
    # DEBUG: confirm model is in train mode
    print(f"DEBUG: train() called for epoch {epoch}; model.training = {model.training}")
    for batch_idx, (data, label) in enumerate(train_loader):
        # ensure tensors on right device and correct dtype for labels
        data = data.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True).long()

        if args.pre_trained:
            # weighted_eps_list expected to be an array-like indexed by batch_idx OR precomputed per-sample
            weighted_eps = torch.from_numpy(weighted_eps_list[f'arr_{batch_idx}']).to(device)
        else:
            # if weighted_eps_list is a python list/np.array of same length as loader
            # ---- start robust weighted_eps selection ----
            weighted_eps = None
            if weighted_eps_list is not None:
                try:
                    # Debug: show lengths once (helps confirm format)
                    if batch_idx == 0:
                        try:
                            print("DEBUG: weighted_eps_list length:", len(weighted_eps_list),
                                  "num_batches:", len(train_loader), "dataset_size:", len(train_loader.dataset))
                        except Exception:
                            pass
            
                    # Case A: weighted_eps_list is provided per-batch
                    if len(weighted_eps_list) == len(train_loader):
                        weighted_eps = weighted_eps_list[batch_idx]
                    else:
                        # Case B: weighted_eps_list likely per-sample -> gather the per-sample entries for this batch
                        # Determine sample indices for this batch in the underlying dataset
                        if hasattr(train_loader.dataset, 'indices'):
                            # When using Subset (your train_loader was replaced), use its indices
                            all_indices = train_loader.dataset.indices
                        else:
                            # Full dataset (no Subset): assume contiguous ordering
                            all_indices = list(range(len(train_loader.dataset)))
            
                        start = batch_idx * train_loader.batch_size
                        end = start + data.size(0)  # data.size(0) handles last smaller batch
                        batch_sample_indices = all_indices[start:end]
            
                        # collect per-sample weighted eps for this batch
                        batch_w = [weighted_eps_list[i] for i in batch_sample_indices]
            
                        # convert to a batched tensor if elements are tensors, else list of tensors
                        if len(batch_w) > 0 and torch.is_tensor(batch_w[0]):
                            weighted_eps = torch.stack([w.to(device) for w in batch_w])
                        else:
                            weighted_eps = [torch.as_tensor(w).to(device) if not torch.is_tensor(w) else w.to(device)
                                            for w in batch_w]
                except Exception as e:
                    print("Warning: could not index weighted_eps_list for batch", batch_idx, ":", repr(e))
                    weighted_eps = None
            # ---- end robust selection ----



        # calculate robust perturbation
        # ---------------- TEMP DEBUG: use clean data ----------------
        # model.eval()
        # if args.attack == 'pgd':
        #     data_adv = part_pgd(model, ...)
        # elif args.attack == 'mma':
        #     data_adv = part_mma(model, ...)
        # else:
        #     raise ValueError("Unknown attack")
        
        # Use clean images (no adversarial attack) for quick debug epoch
        data_adv = data
        # ----------------------------------------------------------


        model.train()
        optimizer.zero_grad()
        out = model(data_adv)
        loss = F.cross_entropy(out, label)
        loss.backward()
        optimizer.step()
        # DEBUG: light optimizer stepping check
        try:
            n_states = sum(1 for k in optimizer.state.keys())
            print(f"DEBUG: after optimizer.step() - optimizer.state entries: {n_states}")
        except Exception as e:
            print("DEBUG: optimizer inspection failed:", repr(e))


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
    start_epoch, _ = try_resume(model, optimizer, model_dir, device)
    # ----------------- DEBUG: resume sanity checks -----------------
    # Print start epoch and basic checkpoint/optimizer info
    print("DEBUG: start_epoch =", start_epoch)
    # Inspect optimizer state (quick, safe)
    try:
        ckpt_path = os.path.join(model_dir, 'latest.pth')
        if os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, map_location='cpu')
            print("DEBUG: ckpt epoch:", ck.get('epoch'))
            print("DEBUG: ckpt keys:", list(ck.keys()))
            # quick optimizer state size check
            st = ck.get('optimizer_state', {}).get('state', {})
            print("DEBUG: optimizer_state entries (sample):", list(st.keys())[:5], "len=", len(st))
        else:
            print("DEBUG: no latest.pth found at", ckpt_path)
    except Exception as e:
        print("DEBUG: failed resume-inspect:", repr(e))
    # ---------------------------------------------------------------

    weighted_eps_path = os.path.join(model_dir, 'weighted_eps_latest.npy')
    weighted_eps_list = safe_load_weighted_eps(weighted_eps_path, model, train_loader, device, args)
    
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

    # ===== DEBUG: train-subset accuracy check (after warm-up) =====
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
            if total >= 500:  # limit to first 500 samples
                break
    print(f"DEBUG: train-subset accuracy (first {total}): {correct}/{total} = {100*correct/total:.2f}%")
    model.train()
    # =============================================================


    # compute or reload weighted_eps_list
    if weighted_eps_list is None:
        weighted_eps_list = save_cam(model, train_loader, device, args)
        # save latest weighted eps list
        def to_cpu_numpy(x):
          if torch.is_tensor(x):
            return x.detach().cpu().numpy()
          return x

        weighted_eps_list_cpu = [to_cpu_numpy(w) for w in weighted_eps_list]

        safe_numpy_save(
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
            # convert to CPU numpy objects for safe storage
            w_cpu = []
            for w in weighted_eps_list:
                if torch.is_tensor(w):
                    w_cpu.append(w.detach().cpu().numpy())
                else:
                    w_cpu.append(np.array(w, dtype=object))
            # atomic saves: per-epoch and latest
            safe_numpy_save(os.path.join(model_dir, f'weighted_eps_epoch{main_epoch}.npy'),
                            np.array(w_cpu, dtype=object))
            safe_numpy_save(os.path.join(model_dir, 'weighted_eps_latest.npy'),
                            np.array(w_cpu, dtype=object))
            print(f"Saved weighted_eps for epoch {main_epoch}")
    
        # adjust learning rate for this epoch
        adjust_learning_rate(args, optimizer, main_epoch)
    
        # adversarial training for this epoch
        train(args, model, device, train_loader, optimizer, main_epoch, weighted_eps_list)
    
        # save checkpoint (per epoch) and latest
        epoch_global = args.warm_up + main_epoch
        save_checkpoint(os.path.join(model_dir, 'latest.pth'), model, optimizer, epoch_global)
        torch.save(model.state_dict(), os.path.join(model_dir, f'part_epoch{epoch_global}.pth'))
        if epoch_global % args.save_freq == 0:
            print('saved model at global epoch', epoch_global)
    
        print('================================================================')


    # evaluation on adversarial examples
    modes = args.eval_modes.split(',')
    modes = [m.strip().lower() for m in modes]
    if 'all' in modes:
        modes = ['pgd','mma','aa']

    # ----------------- DEBUG: quick one-batch sanity check on test loader -----------------
    print("DEBUG: running one-batch sanity check on test_loader")
    model.eval()
    try:
        imgs, labels = next(iter(test_loader))
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            preds = model(imgs).argmax(dim=1)
        print("DEBUG: labels[:20]:", labels[:20].cpu().numpy())
        print("DEBUG: preds [:20]:", preds[:20].cpu().numpy())
        # quick counts
        unique_preds = torch.unique(preds).cpu().numpy().tolist()
        print("DEBUG: unique preds in batch (sample):", unique_preds)
    except Exception as e:
        print("DEBUG: one-batch sanity check failed:", repr(e))
    # --------------------------------------------------------------------------------------

    for mode in modes:
        if mode not in ('pgd','mma','aa'):
            print("Unknown eval mode:", mode)
            continue
        print(f'{mode.upper()}=============================================================')
        eval_test(args, model, device, test_loader, mode=mode)



if __name__ == '__main__':
    main()





















