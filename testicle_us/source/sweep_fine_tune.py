# sweeps_fine_tune.py
import os
import sys
import wandb
import argparse
from utils import *
from dataset import *
from net import *
from trainer import *


def main():
    # Initialize wandb run
    with wandb.init() as run:
        # Get hyperparameters from wandb.config
        config = wandb.config

        # Create argparse Namespace from wandb config
        args = argparse.Namespace(**{k: v for k, v in config.items()})

        # Print args
        print("Running with configuration:")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")

        all_metrics = []
        for seed in args.seeds:
            device = load_device(args.device)
            dataset = USDataset(root_dir=args.dataset_dir, mode=args.dataset_mode)
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

        all_accuracy = torch.tensor([m['accuracy']['values'] for m in all_metrics]).reshape(-1)
        all_precision = torch.tensor([m['precision']['mean'] for m in all_metrics])
        all_recall = torch.tensor([m['recall']['mean'] for m in all_metrics])
        all_f1 = torch.tensor([m['f1_score']['mean'] for m in all_metrics])
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
        print("\n" + "=" * 50)
        print("FINAL RESULTS ACROSS ALL SEEDS")
        print("=" * 50)
        for metric_name, values in final_metrics.items():
            print(f"{metric_name}: {values['mean']:.4f} ± {values['std']:.4f}")

        for metric_name, values in final_metrics.items():
            wandb.config.update({
                f"{metric_name}_mean": f"{values['mean']:.4f}",
                f"{metric_name}_std": f"{values['std']:.4f}"
            })
            wandb.log({
                f"{metric_name}_mean": f"{values['mean']:.4f}",
                f"{metric_name}_std": f"{values['std']:.4f}"
            })


if __name__ == "__main__":
    main()
    sys.exit(0)
