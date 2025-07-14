import os, sys
from utils import *
from torchvision.datasets import ImageFolder
from net import *
from trainer import *
import argparse
from PIL import Image


def parse_args(defaults: dict):
    parser = argparse.ArgumentParser()

    # DIRECTORIES config
    parser.add_argument("--checkpoint_dir", type=str, help="Path in which checkpoint shall be saved")
    parser.add_argument("--dataset_dir", type=str, help="Path containing the dataset")
    parser.add_argument("--backbone", choices=[None, "rn18", "rn50", "uscl", "resnet_fpn", "resnet_fpn_hico", "densenet"],
                        type=str, default=None, help="Backbone type to be used")
    parser.add_argument("--load_backbone", type=str, default=None, help="Backbone weights path to be loaded")

    # TRAINING OPTIONS
    exclusive_group = parser.add_mutually_exclusive_group(required=True)
    exclusive_group.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    exclusive_group.add_argument("--steps", type=int, default=None, help="Number of training steps")

    parser.add_argument("--batch_size", type=int, help="batch size of dataloader")
    parser.add_argument("--img_ch", type=int, default=1, help="Number of channles of images")
    parser.add_argument("--opt", type=str, choices=["adam", "lars",], help="name of teh optimizer to be used")
    parser.add_argument("--lr", type=float, help="initial learning rate to be used")
    parser.add_argument("--sched", type=str, choices=["poly"], help="name of the scheduler to be used")
    parser.add_argument("--use_sup_loss", action="store_true", help="to use sup loss")
    parser.add_argument("--use_unsup_loss", action="store_true", help="to use unsup loss")
    parser.add_argument("--warmup", action="store_true", help="to set if want warmup")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="n warmup epochs")
    parser.add_argument("--norm_mean", type=float, default=0.5, help="mean to be used for normalization")
    parser.add_argument("--norm_std", type=float, default=0.25, help="std to be used for normalization")
    parser.add_argument("--lambda_", type=float, default=0.2, help="lambda to be used for unsup_loss+lambda_*sup_loss")
    # HARDWARE
    parser.add_argument("--device", type=str, choices=["cpu", "gpu",], help="device where the computation will happend")
    parser.add_argument("--data_parallel", action="store_true", help="to set if data parallel should be used")
    parser.add_argument("--num_workers", type=int, default=1, help="number of workers that can be created")

    # WNADB CONFIG
    parser.add_argument("--project_name", type=str, help="wandb project name")
    parser.add_argument("--run_name", type=str, help="wandb run name")

    parser.set_defaults(**defaults)
    return parser.parse_args()


def fine_tune_defauls():
    return dict(
        checkpoint_dir=f"{str(Path(__file__).parent)}/checkpoint",
        batch_size=256,
        opt="adam",
        lr=0.005,
        sched="poly",
        project_name="tesi_prj2_backbones",
    )


def main():
    args = parse_args(fine_tune_defauls())
    display_args(args)
    device = load_device(args.device)
    dataset = ImageFolder(root=args.dataset_dir, transform=v2.Compose([
        v2.ToImage(),
        v2.Resize((256, 256))]))
    net = BackboneWrapper.create_backbone(num_classes=len(dataset.classes),
                                          backbone=args.backbone, weights_path=args.load_backbone)
    opt = load_opt(args, net)
    sched = load_sched(args, opt)

    tr = ContrastiveTrainer(
        net=net,
        device=device,
        args=args,
        opt=opt,
        scheduler=sched
    )
    tr.train(dataset)


if __name__ == "__main__":
    main()
    sys.exit(0)
