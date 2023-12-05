import argparse
import torch
import datetime
import json
import yaml
import os

from D3M_main_model import TS_D3M_Physio
from dataset_physio import get_dataloader
from utils import train_D3M, evaluate_D3M

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="ddm_base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--save", type=int, default=1)
parser.add_argument('--num_steps', type=int, default=10)
parser.add_argument("--ddm_type", type=str, default='constant')
parser.add_argument('--beta_type', type=str, default='sqrt')
parser.add_argument('--time_move', type=int, default=1)
parser.add_argument('--feat_move', type=int, default=1)

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio
config['model']['time_move'] = args.time_move
config['model']['feat_move'] = args.feat_move

# 新增内容
config['train']['epochs'] = args.epochs
config['train']['batch_size'] = args.bs
config['diffusion']['num_steps'] = args.num_steps
config['diffusion']['ddm_type'] = args.ddm_type
config['diffusion']['beta_type'] = args.beta_type

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/physio_fold" + str(args.nfold) + "_" + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader = get_dataloader(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
)

model = TS_D3M_Physio(config, args.device).to(args.device)

if args.modelfolder == "":
    train_D3M(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))
    print('load model success')

if config['diffusion']['ddm_type'] == 'linear':
    clamp = True
    use_pred = True
else:
    clamp = False
    use_pred = False
evaluate_D3M(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername,
             use_pred=use_pred, denoise=True, clamp=clamp, config=config, save=bool(args.save))
