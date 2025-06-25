import argparse
import pickle as pk

import numpy as np
import pandas as pd
import torch
from e3nn.io import CartesianTensor
from jarvis.core.atoms import Atoms
from pandarallel import pandarallel
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.jarvis import JarvisAtomsAdaptor
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import get_dataset

pandarallel.initialize(progress_bar=False)

from e3nn import o3
from graphs import GraphDataset, atoms2graphs
from optixnet import OptixNet
from utils import fix_outputs_by_crystal_type, get_id_train_val_test

# torch config
torch.set_default_dtype(torch.float32)
import os
import random

import numpy as np
import torch

# Set the random seed for Python, NumPy, and PyTorch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

# Ensuring CUDA's determinism
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # if using multi-GPU.
    # Configure PyTorch to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

adptor = JarvisAtomsAdaptor()

diagonal = [0, 4, 8]
off_diagonal = [1, 2, 3, 5, 6, 7]
converter = CartesianTensor("ij")
irreps_output = o3.Irreps('1x0e + 1x0o + 1x1e + 1x1o + 1x2e + 1x2o + 1x3e + 1x3o')

def structure_to_graphs(
    df: pd.DataFrame,
    reduce_cell: bool = False,
    cutoff: float = 4.0,
    max_neighbors: int = 16
):
    def atoms_to_graph(p_input):
        """Convert structure dict to DGLGraph."""
        structure = adptor.get_atoms(p_input["structure"])
        return atoms2graphs(
            structure,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            reduce=reduce_cell,
            equivalent_atoms=p_input['equivalent_atoms'],
            use_canonize=True,
        )
    graphs = df["p_input"].parallel_apply(atoms_to_graph).values
    # graphs = df["p_input"].apply(atoms_to_graph).values
    return graphs

def count_parameters(model):
    total_params = 0
    for parameter in model.parameters():
        total_params += parameter.element_size() * parameter.nelement()
    for parameter in model.buffers():
        total_params += parameter.element_size() * parameter.nelement()
    total_params = total_params / 1024 / 1024
    print(f"Total size: {total_params}")
    print("Total trainable parameter number", sum(p.numel() for p in model.parameters() if p.requires_grad))
    return total_params


class PolynomialLRDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iters, start_lr, end_lr, power=1, last_epoch=-1):
        self.max_iters = max_iters
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.power = power
        self.last_iter = 0  # Custom attribute to keep track of last iteration count
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            (self.start_lr - self.end_lr) * 
            ((1 - self.last_iter / self.max_iters) ** self.power) + self.end_lr 
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        self.last_iter += 1  # Increment the last iteration count
        return super().step(epoch)

def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def get_pyg_dataset(data, target, reduce_cell=False, add_formulas=False):
    df_dataset = pd.DataFrame(data)
    g_dataset = structure_to_graphs(df_dataset, reduce_cell=reduce_cell)
    pyg_dataset = GraphDataset(df=df_dataset,graphs=g_dataset, target=target, add_formuals=add_formulas)
    return pyg_dataset


def load_dataloaders(args):
    if args.load_preprocessed:
        print("load preprocessed dataset ...")
    dataset_sym = get_dataset(
        raw_dataset_path=args.raw_data,
        path2save=args.path2save_data,
        dataset_name=args.target,
        load_preprocessed=args.load_preprocessed
    )
    # pdb.set_trace()
    # preprocess the dataset and random split
    id_train, id_val, id_test = get_id_train_val_test(
            total_size=len(dataset_sym),
            split_seed=args.split_seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            keep_data_order=False,
        )

    dataset_train = [dataset_sym[x] for x in id_train]
    dataset_val = [dataset_sym[x] for x in id_val]
    dataset_test = [dataset_sym[x] for x in id_test]

    dataset_train = [d for d in dataset_train if d['crystal_class'] not in ['222', '4-2m', '23', '4-3m']]
    dataset_val = [d for d in dataset_val if d['crystal_class'] not in ['222', '4-2m', '23', '4-3m']]
    dataset_test = [d for d in dataset_test if d['crystal_class'] not in ['222', '4-2m', '23', '4-3m']]

    pyg_dataset_train = get_pyg_dataset(dataset_train, args.target, args.reduce_cell)
    pyg_dataset_val = get_pyg_dataset(dataset_val, args.target, args.reduce_cell)
    pyg_dataset_test = get_pyg_dataset(dataset_test, args.target, args.reduce_cell)

    # form dataloaders
    collate_fn = pyg_dataset_train.collate
    train_loader = DataLoader(
        pyg_dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        pyg_dataset_val,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        pyg_dataset_test,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(args, model, train_loader, optimizer, scheduler, criterion, epoch):
    model.train()
    running_loss = 0.0
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{args.epochs}", unit='batch') as pbar:
        for l, data in enumerate(train_loader):
            structure, mask, equality, labels, crystal_type = data
            structure, mask, equality, labels = structure.to(device), mask.to(device), equality.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(structure)
            for i in range(len(outputs)):
                outputs[i] = fix_outputs_by_crystal_type(outputs[i], crystal_type[i])
   
            outputs_max = torch.max(torch.abs(outputs), dim=-1).values
            labels_max = torch.max(torch.abs(labels), dim=-1).values
            loss_max = torch.sqrt(criterion(outputs_max, labels_max))
            loss_tensor = criterion(outputs, labels)

            loss = loss_tensor
            # loss = loss_tensor * 0.3 + loss_max for second stage training

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'training_loss': running_loss / (pbar.n + 1)})
            pbar.update(1)
            scheduler.step()


def validate_model(model, val_loader):
    model.eval()
    label_list = []
    output_list = []
        
    for data in val_loader:
        structure, mask, _, labels, crystal_type = data
        structure, mask, labels = structure.to(device), mask.to(device), labels.to(device)
        outputs = model(structure).detach()
        for i in range(len(outputs)):
            outputs[i] = fix_outputs_by_crystal_type(outputs[i], crystal_type[i])

        output_list.append(outputs.reshape(-1, 18))

        label_list.append(labels.reshape(-1, 18))

        
    outputs = torch.stack(output_list).reshape(-1, 3, 6)
    labels = torch.stack(label_list).reshape(-1, 3, 6)
    outputs = torch.max(torch.abs(outputs), dim=-1).values
    labels = torch.max(torch.abs(labels), dim=-1).values
    mae = abs(outputs - labels).mean(dim=-1).mean()

    return mae


def train(model, args):
    # load the dataset
    train_loader, val_loader, test_loader = load_dataloaders(args)
    print("n_train:", len(train_loader.dataset))
    print("n_val:", len(val_loader.dataset))
    print("n_test:", len(test_loader.dataset))

    # set up training configs
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = len(train_loader)
    total_iter = steps_per_epoch * args.epochs
    scheduler = PolynomialLRDecay(optimizer, max_iters=total_iter, start_lr=args.learning_rate, end_lr=0.00001, power=1)
    
    criteria = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(reduction='mean'),
        "huber": nn.HuberLoss()
    }
    criterion = criteria[args.loss]
 
    # training epoch
    best_score = 10000
    for epoch in range(args.epochs):
        train_one_epoch(
            args=args,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            epoch=epoch
        )
        # Validation
        mae = validate_model(model=model, val_loader=val_loader)
        if mae < best_score:
            best_score = mae
            torch.save(model.state_dict(), "runs/%s/model_best.pt"%(args.name))

        print("Validation mae ", mae)

    torch.save(model.state_dict(), "runs/%s/final_model_test_corrected%s.pt"%(args.name, args.model))
    return

def main():
    parser = argparse.ArgumentParser(description='Training script')

    # Define command-line arguments
    # training parameters
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of training and evaluating')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-05, help='weight decay')
    parser.add_argument('--loss', type=str, default='huber', help='mse or l1 or huber')
    parser.add_argument('--model', type=str, default='comformer', help='comformer or megnet')
    parser.add_argument('--load_model', type=bool, default=False, help='load pretrained model or not')
    parser.add_argument('--name', type=str, default='test', help='name of project for storage')
    # dataset parameters
    parser.add_argument('--split_seed', type=int, default=32, help='the random seed of spliting data')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='training ratio used in data split')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='evaluate ratio used in data split')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='test ratio used in data split')
    parser.add_argument('--target', type=str, default='shg', help='dielectric, piezoelectric, or elastic')
    parser.add_argument('--load_preprocessed', type=bool, default=False, help='load previous processed dataset')
    parser.add_argument('--raw_data', type=str)
    parser.add_argument('--path2save_data', type=str)
    parser.add_argument("--model_path", type=str, default="")

    args = parser.parse_args()

    print('Training settings:')
    print(f'  Epochs: {args.epochs}')
    print(f'  Learning rate: {args.learning_rate}')
    print(args)
    torch.manual_seed(args.split_seed)
    torch.cuda.manual_seed_all(args.split_seed)
    # load the model
    if args.model == "optixnet":
        model = OptixNet(args)
    else:
        raise NotImplementedError(f"Unknown model: {args.model}")

    if not os.path.exists('runs/' + args.name):
        # Create the directory
        os.makedirs('runs/' + args.name)
        
    if args.load_model:
        saved_model_path = args.model_path
        state_dict = torch.load(saved_model_path)
        model.load_state_dict(state_dict)

    train(model, args)

if __name__ == "__main__":
    main()