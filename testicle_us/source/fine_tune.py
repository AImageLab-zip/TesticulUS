import os, sys
from utils import *
from dataset import *
from net import *
from trainer import *
import argparse

def label_type_parser(value):
    """Custom type function to convert string to LabelTypes enum"""
    label_map = {
        "omo_disomo": LabelTypes.OMO_DISOMO,
        "binary_functional": LabelTypes.BINARY_FUNCTIONAL,
        "functional": LabelTypes.FUNCTIONAL
    }
    
    if value.lower() in label_map:
        return label_map[value.lower()]
    else:
        raise argparse.ArgumentTypeError(f"Invalid label_type: {value}. Choose from {list(label_map.keys())}")



def parse_args(defaults: dict):
    parser = argparse.ArgumentParser()

    # MODEL config
    parser.add_argument("--backbone", choices=[None, "uscl", "rn18", "rn50", "hico", "resnet_fpn", "densenet"],
                        type=str, default=None, help="Backbone type to be used")
    parser.add_argument("--load_backbone", type=str, default=None, help="Backbone weights path to be loaded")

    # DIRECTORIES config
    parser.add_argument("--checkpoint_dir", type=str, help="Path in which checkpoint shall be saved")
    parser.add_argument("--logs_dir", type=str, help="Path in which logs shall be saved")
    parser.add_argument("--dataset_dir", type=str, help="Path containing the dataset")
    parser.add_argument("--excluded_ids", default=None, type=str, help="Path containing the list of ids to exclude")
    parser.add_argument("--flipped_ids", default=None, type=str, help="Path containing the list of ids to flip")

    # TRAINING OPTIONS
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--img_ch", type=int, default=1, help="Number of channles of images")
    parser.add_argument("--n_folds", type=int, help="Number of folds")
    parser.add_argument("--batch_size", type=int, help="batch size of dataloader")
    parser.add_argument("--opt", type=str, choices=["adam",], help="name of teh optimizer to be used")
    parser.add_argument("--lr_back", type=float, help="initial learning rate to be used on backbone")
    parser.add_argument("--lr_head", type=float, help="initial learning rate to be used on head")
    parser.add_argument("--sched", type=str, choices=[None, "poly"], help="name of the scheduler to be used")
    parser.add_argument("--num_classes", type=int, help="number of classes in the classification head")
    parser.add_argument("--augment", action="store_true", help="if setted augmentation will be used")
    parser.add_argument("--save_check", action="store_true", help="if setted checkpoint will be saved")
    parser.add_argument("--weighted_sampler", action="store_true", help="if setted weighted sampler will be used")
    parser.add_argument("--mask_outliers", action="store_true",
                        help="if setted train examples which looks like outliers by ema will be removed")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="if setted the backbone during the finetuning will be froozen")
    parser.add_argument("--save_losses_dict", action="store_true",
                        help="if setted dict containing the losses over epoch of every example will be saved in log dir")
    parser.add_argument('-s', '--seeds', nargs='+', type=int,
                        help='A list of integers (e.g., --seeds 1 2 3)')
    parser.add_argument("--dataset_mode", type=str,
                        choices=[None, "single_view", "coupled_view"], help="how images are preprocessed by dataset class")
    parser.add_argument("--norm_mean", type=float, default=0.5, help="mean to be used for normalization")
    parser.add_argument("--norm_std", type=float, default=0.25, help="std to be used for normalization")
    parser.add_argument("--wandb_log", action="store_true", help="if setted the run will be logged on wandb")
    parser.add_argument("--label_type", type=label_type_parser, default=LabelTypes.OMO_DISOMO,
                        help="Type of labels to use. Options: omo_disomo, binary_functional, functional (default: omo_disomo)")

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
        logs_dir=f"{str(Path(__file__).parent)}/.log",
        epochs=30,
        batch_size=64,
        opt="adam",
        lr=0.0001,
        sched=None,
        num_classes=2,
        project_name="tesi_prj3",
        augment=False,
        dataset_mode="single_view",
        n_folds=3
    )


def aggregate_and_log_results(all_metrics, args):
    # Extract metrics across all seeds
    all_accuracy = torch.tensor([m['accuracy']['values'] for m in all_metrics]).reshape(-1)
    all_precision = torch.tensor([m['precision']['mean'] for m in all_metrics])
    all_recall = torch.tensor([m['recall']['mean'] for m in all_metrics])
    all_f1 = torch.tensor([m['f1_score']['mean'] for m in all_metrics])

    # Calculate mean and std across seeds
    final_metrics = {
        "accuracy": {
            "mean": torch.mean(all_accuracy).item(),
            "std": torch.std(all_accuracy).item()
        },
        "precision": {
            "mean": torch.mean(all_precision).item(),
            "std": torch.std(all_precision).item()
        },
        "recall": {
            "mean": torch.mean(all_recall).item(),
            "std": torch.std(all_recall).item()
        },
        "f1": {
            "mean": torch.mean(all_f1).item(),
            "std": torch.std(all_f1).item()
        }
    }

    # Print final results
    print("\n" + "=" * 50)
    print("FINAL RESULTS ACROSS ALL SEEDS")
    print("=" * 50)
    for metric_name, values in final_metrics.items():
        print(f"{metric_name}: {values['mean']:.4f} ± {values['std']:.4f}")

    # Log to wandb if needed
    import wandb
    if not wandb.api.api_key:
        print("Wandb API key not set, skipping logging")
        return
    wandb.init(
        project="tesi_prj2_metrics",
        # name=f"{args.run_name}_{args.load_backbone.split('_')[-1]}",
        name=f"{args.run_name}",
        config=args.__dict__
    )
    # Log the seed-level results
    for i, metrics in enumerate(all_metrics):
        for metric_type in ['accuracy', 'precision', 'recall', 'f1_score']:
            wandb.log({
                f"seed_{metrics['seed']}_{metric_type}_mean": metrics[metric_type]['mean'],
                f"seed_{metrics['seed']}_{metric_type}_std": metrics[metric_type]['std']
            })
    # Log the final aggregated results
    for metric_name, values in final_metrics.items():
        wandb.config.update({
            f"{metric_name}_mean": f"{values['mean']:.4f}",
            f"{metric_name}_std": f"{values['std']:.4f}"
        })
    wandb.finish()


def main():
    args = parse_args(fine_tune_defauls())
    display_args(args)

    all_metrics = []
    for seed in args.seeds:
        device = load_device(args.device)
        dataset = USDataset(root_dir=args.dataset_dir, mode=args.dataset_mode, label_type=args.label_type)
        if args.label_type == LabelTypes.OMO_DISOMO:
            if args.excluded_ids is not None:
                remove_bad_ids(dataset, args.excluded_ids)
            if args.flipped_ids is not None:
                flip_bad_ids(dataset, args.flipped_ids)

        net = ClassificationWrapper(backbone=args.backbone,
                                    num_classes=args.num_classes, weights_path=args.load_backbone)
        opt = load_opt(args, net)
        sched = load_sched(args, opt)
        tr = ClassificationTrainer(
            net=net,
            device=device,
            args=args,
            opt=opt,
            scheduler=sched,
            seed=seed,
        )

        if args.n_folds > 1:
            metrics = tr.train_cv(dataset)
        else:
            metrics = tr.train_split(dataset)

        metrics['seed'] = seed
        all_metrics.append(metrics)

    if args.wandb_log:
        aggregate_and_log_results(all_metrics, args)


if __name__ == "__main__":
    main()
    sys.exit(0)
