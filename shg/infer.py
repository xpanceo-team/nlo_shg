
import json
from argparse import ArgumentParser

import torch
import tqdm
from optixnet import OptixNet
from torch.utils.data import DataLoader
from train import get_pyg_dataset
from utils import fix_outputs_by_crystal_type

from data import get_dataset

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

def load_data(args):
    dataset_sym = get_dataset(
        raw_dataset_path=args.raw_data,
        path2save=args.path2save_data,
        dataset_name=args.target,
        load_preprocessed=args.load_preprocessed,
    )
    

    dataset_test = dataset_sym
    
    pyg_dataset_test = get_pyg_dataset(dataset_test, args.target)
    

    collate_fn = pyg_dataset_test.collate

    test_loader = DataLoader(
        pyg_dataset_test,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )
    print("n_test:", len(test_loader.dataset))
    
    return test_loader

def infer(args, model):
    test_loader = load_data(args)
    
    model.eval()
    output_list = []
    
    for data in tqdm.tqdm(test_loader):
        structure, _, _, _, crystal_type = data
        structure = structure.to(device)
        outputs = model(structure).detach()
        for i in range(len(outputs)):
            outputs[i] = fix_outputs_by_crystal_type(outputs[i], crystal_type[i])

        output_list.append(outputs.reshape(-1, 18))

    outputs = torch.stack(output_list).reshape(-1, 18)
    
    outputs = outputs.detach().cpu().numpy().tolist()

    with open(args.output_path, "w") as f:
        json.dump(outputs, f)
    
    return outputs


def parse_args():
    parser = ArgumentParser(description='Inference script')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of training and evaluating')
    parser.add_argument('--model', type=str, default='comformer', help='comformer or megnet')
    parser.add_argument('--load_model', action='store_true', help='load pretrained model or not')
    parser.add_argument('--output_path', type=str, help="Path to store output in json format")
    parser.add_argument('--target', type=str, default='shg')
    parser.add_argument('--load_preprocessed', type=bool, default=False, help='load previous processed dataset')
    parser.add_argument('--raw_data', type=str)
    parser.add_argument('--path2save_data', type=str)
    parser.add_argument("--model_path", type=str, default="")

    return parser.parse_args()

def main(args):

    if args.model == "optixnet":
        model = OptixNet(args)
    else:
        raise NotImplementedError(f"Unknown model: {args.model}")

        
    if args.load_model:
        saved_model_path = args.model_path
        state_dict = torch.load(saved_model_path)
        model.load_state_dict(state_dict)

    model.to(device)
    infer(args, model)

if __name__ == "__main__":
    args = parse_args()
    main(args)