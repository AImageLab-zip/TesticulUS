from functools import partial
import os, sys, wandb
import wandb.plot
from pprint import pprint
from hico.loss import *
from net import BackboneWrapper
import utils
from dataset import custom_collate as custom_collate_dataset
from utils import *
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn as nn
import torch, random
from torch.utils.data import WeightedRandomSampler
import numpy as np
from sklearn.model_selection import KFold
import copy
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from contextlib import nullcontext
from sklearn.model_selection import train_test_split
from torch.nn import DataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def seed_worker(worker_id):
    worker_seed = int(os.environ["SEED"])
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def custom_collate(batch, norm_mean=0.5, norm_std=0.25, img_ch=1):
    data_aug_lhs = v2.Compose(
        [
            v2.RandomResizedCrop(size=256, scale=(0.8, 1.0), ratio=(0.8, 1.25)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.Grayscale(num_output_channels=img_ch),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=[norm_mean], std=[norm_std]),
        ]
    )

    template_paths = [
        f"{str(Path(__file__).parent)}/O_marker.png",
    ]
    thresholds = [0.8]
    data_aug_rhs = v2.Compose(
        [
            TemplateMatchingTransform(
                template_paths=template_paths, thresholds=thresholds
            ),
            RandomShadowSquare(shadow_size=96, p=0.7),
            v2.RandomRotation(90),
            v2.RandomHorizontalFlip(),
            v2.RandomApply(
                [
                    v2.ColorJitter(0.8, 0.8, 0.8, 0.2),
                ],
                p=0.8,
            ),
            v2.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            v2.Grayscale(num_output_channels=img_ch),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=[norm_mean], std=[norm_std]),
        ]
    )
    # data_aug = DataAugmentation()

    images, labels = zip(*batch)  # unzips list of tuples into two tuples
    images = torch.stack(images)
    labels = torch.tensor(labels)

    augmented_images_1 = data_aug_lhs(images)
    augmented_images_2 = data_aug_rhs(images)

    return images, augmented_images_1, augmented_images_2, labels


def custom_collate_fake_dataset(
    batch,
    norm_mean=0.5,
    norm_std=0.25,
    img_ch=1,
    use_augmentation=False,
    return_ids=False,
):
    images, labels = zip(*batch)
    if use_augmentation:
        aug = utils.FineTuningDataAug_v2()
    std_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((256, 256)),
            v2.Grayscale(num_output_channels=img_ch),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=[norm_mean], std=[norm_std]),
        ]
    )
    transformed_imgs = []
    for img in images:
        if use_augmentation:
            img = aug(img)
        transformed_imgs.append(std_transform(img))

    if return_ids:
        return (
            torch.stack(transformed_imgs),
            torch.tensor(labels),
            torch.zeros(len(transformed_imgs)),
        )
    else:
        return torch.stack(transformed_imgs), torch.tensor(labels)


class ClassificationTrainer:
    def __init__(self, net, device, args: Namespace, opt, scheduler, seed):
        self.net = net
        self.device = device
        self.args = args
        self.opt = opt
        self.scheduler = scheduler
        self.seed = seed
        os.environ["SEED"] = str(self.seed)
        print(f"Used seed: {self.seed}")
        self.gen = torch.Generator().manual_seed(self.seed)
        self.set_seed(self.seed)
        self.loss = nn.functional.cross_entropy

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def init_wandb(self, placeholder):
        assert wandb.api.api_key, "the api key has not been set!\n"
        wandb.login(verify=True)
        wandb.init(
            project=self.args.project_name,
            name=f"{self.args.run_name}_{placeholder}",
            config=self.args.__dict__,
        )
        wandb.config.update({"seed": self.seed})

    def process_epoch_(
        self,
        net,
        dataloader,
        device,
        epoch,
        fold_n,
        phase="train",
        mask_dict: dict = None,
    ):

        net = net.to(device)
        running_loss = 0.0
        contrastive_sums = 0.0
        all_preds = []
        all_targets = []
        example_losses = {}
        total_masked_examples = 0
        excluded_ids = []
        excluded_examples = 0

        # Set model mode
        if phase == "train":
            net.train()
        else:  # eval
            net.eval()

        pbar = tqdm(dataloader, desc=f"{phase}-{epoch}")
        context = torch.no_grad() if phase == "eval" or phase == "val" else nullcontext()

        with context:
            for i, (x_batch, y_batch, ids) in enumerate(dataloader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).type(torch.int64)
                batch_size = x_batch.shape[0]
                outputs = net(x_batch)
                # loss = self.loss(input=outputs.float(), target=y_batch.long())
                individual_losses = self.loss(
                    input=outputs.float(), target=y_batch.long(), reduction="none"
                )

                for j in range(batch_size):
                    if ids[j] in example_losses.keys():
                        example_losses[ids[j]].append(individual_losses[j])
                    else:
                        example_losses[ids[j]] = [individual_losses[j]]

                if mask_dict is not None:
                    masked_loss = 0
                    for j in range(batch_size):
                        if ids[j] not in mask_dict.keys() or mask_dict[ids[j]]:
                            masked_loss += individual_losses[j]
                            total_masked_examples += 1
                        else:
                            excluded_examples += 1
                            excluded_ids.append(ids[j])

                    loss = masked_loss / torch.tensor(
                        total_masked_examples,
                        dtype=torch.float,
                        device=self.device,
                        requires_grad=True,
                    )
                else:
                    loss = individual_losses.mean()

                if self.args.dataset_mode == "coupled_view":
                    contrastive_loss = coupled_loss(outputs, temperature=1)
                    contrastive_sums += contrastive_loss
                    loss += 1 * contrastive_loss

                # Backward pass and optimization (only in training phase)
                if phase == "train":
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    # Log batch loss during training
                    if self.args.wandb_log:
                        wandb.log({f"{phase}/batch_loss": loss.item(), "fold": fold_n})

                # Accumulate loss for epoch average
                running_loss += loss.item()

                # Get predicted class
                _, preds = torch.max(outputs, 1)

                # Store predictions and targets
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"Loss": loss.item()})

        pbar.close()

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        if self.args.num_classes == 2:
            metrics = compute_binary_metrics(all_targets, all_preds)
        else:
            metrics = compute_multiclass_metrics(
                all_targets, all_preds, average="macro", return_per_class=True
            )
        metrics["loss"] = running_loss / len(dataloader)
        metrics["coupled_loss"] = contrastive_sums / len(dataloader)
        metrics["y_true"] = all_targets
        metrics["y_pred"] = all_preds

        if self.args.wandb_log:
            log_metrics(metrics, phase=phase, fold_n=fold_n, epoch=epoch)

        # example_losses = [l.item() for l in example_losses]
        if phase == "train" and self.args.mask_outliers:
            print(f"During epoch-{epoch}:")
            print(f"{excluded_examples} examples have been excluded from loss")
            print(f"the following ids have been discarded")
            pprint(set(excluded_ids))
        return metrics, example_losses

    def train_network_(self, train_net, train_dataloader, test_dataloader, fold_n):
        # ============= Training Loop ===================

        self.opt = utils.load_opt(self.args, train_net)
        self.scheduler = utils.load_sched(self.args, self.opt)

        train_net = train_net.to(self.device)

        if self.args.data_parallel:
            train_net = nn.DataParallel(train_net)

        print("Starting training...", flush=True)
        train_losses_per_epoch = {}
        eval_losses_per_epoch = {}
        for epoch in range(int(self.args.epochs)):
            if self.args.mask_outliers and epoch > 4:
                m, m_shifted = mean_dict(train_losses_per_epoch, 3, 1)
                mask = {}
                for key in m:
                    mask[key] = not (m[key] * 0.8 > m_shifted[key])
            else:
                mask = None
            metrics_train, example_losses_train = self.process_epoch_(
                net=train_net,
                dataloader=train_dataloader,
                device=self.device,
                epoch=epoch,
                fold_n=fold_n,
                phase="train",
                mask_dict=mask,
            )
            metrics_eval, example_losses_eval = self.process_epoch_(
                net=train_net,
                dataloader=test_dataloader,
                device=self.device,
                epoch=epoch,
                fold_n=fold_n,
                phase="eval",
            )
            train_losses_per_epoch = update_dictionary(
                train_losses_per_epoch, example_losses_train
            )
            eval_losses_per_epoch = update_dictionary(
                eval_losses_per_epoch, example_losses_eval
            )

            if self.scheduler is not None:
                self.scheduler.step()
                last_lr = self.opt.param_groups[0]["lr"]
                if self.args.wandb_log:
                    wandb.log({"LR": last_lr})

        if self.args.save_losses_dict:
            torch.save(
                {
                    "train_losses": train_losses_per_epoch,
                    "eval_losses": eval_losses_per_epoch,
                },
                f"{self.args.logs_dir}/{self.args.project_name}_{self.args.run_name}_fold{fold_n}_seed{self.seed}_losses",
            )

        return train_net

    def train_network_with_val(self, train_net, train_dataloader, val_dataloader, test_dataloader, fold_n):
        # ============= Training Loop ===================

        self.opt = utils.load_opt(self.args, train_net)
        self.scheduler = utils.load_sched(self.args, self.opt)

        train_net = train_net.to(self.device)

        if self.args.data_parallel:
            train_net = nn.DataParallel(train_net)

        print("Starting training...", flush=True)
        train_losses_per_epoch = {}
        val_losses_per_epoch = {}
        eval_losses_per_epoch = {}
        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(int(self.args.epochs)):
            if self.args.mask_outliers and epoch > 4:
                m, m_shifted = mean_dict(train_losses_per_epoch, 3, 1)
                mask = {}
                for key in m:
                    mask[key] = not (m[key] * 0.8 > m_shifted[key])
            else:
                mask = None

            # Training phase
            metrics_train, example_losses_train = self.process_epoch_(
                net=train_net,
                dataloader=train_dataloader,
                device=self.device,
                epoch=epoch,
                fold_n=fold_n,
                phase="train",
                mask_dict=mask,
            )

            # Validation phase
            metrics_val, example_losses_val = self.process_epoch_(
                net=train_net,
                dataloader=val_dataloader,
                device=self.device,
                epoch=epoch,
                fold_n=fold_n,
                phase="val",
            )

            # Test phase (optional, can be run only at the end if desired)
            metrics_eval, example_losses_eval = self.process_epoch_(
                net=train_net,
                dataloader=test_dataloader,
                device=self.device,
                epoch=epoch,
                fold_n=fold_n,
                phase="eval",
            )

            train_losses_per_epoch = update_dictionary(
                train_losses_per_epoch, example_losses_train
            )
            val_losses_per_epoch = update_dictionary(
                val_losses_per_epoch, example_losses_val
            )
            eval_losses_per_epoch = update_dictionary(
                eval_losses_per_epoch, example_losses_eval
            )

            # Save best model based on validation loss
            if metrics_val["loss"] < best_val_loss:
                best_val_loss = metrics_val["loss"]
                best_model_state = copy.deepcopy(train_net.state_dict())

            if self.scheduler is not None:
                self.scheduler.step()
                last_lr = self.opt.param_groups[0]["lr"]
                if self.args.wandb_log:
                    wandb.log({"LR": last_lr})

        if self.args.save_losses_dict:
            torch.save(
                {
                    "train_losses": train_losses_per_epoch,
                    "val_losses": val_losses_per_epoch,
                    "eval_losses": eval_losses_per_epoch,
                },
                f"{self.args.logs_dir}/{self.args.project_name}_{self.args.run_name}_fold{fold_n}_seed{self.seed}_losses",
            )

        # Load best model before returning
        if best_model_state is not None:
            train_net.load_state_dict(best_model_state)

        return train_net

    def train_cv(self, dataset):
        # ============= Preparing dataset... ==================
        self.dataset = dataset
        workers_ = self.args.num_workers
        if self.args.freeze_backbone:
            self.net.freeze_backbone()
            print("Backbone has been freezed")

        kf = StratifiedKFold(
            n_splits=self.args.n_folds, shuffle=True, random_state=self.seed
        )

        folds_accuracies = []
        folds_precisions = []
        folds_recalls = []
        f1_scores = []

        for fold, (train_idx, val_idx) in enumerate(
            kf.split(X=list(zip(*dataset.items))[0], y=list(zip(*dataset.items))[1])
        ):
            if self.args.wandb_log:
                self.init_wandb(f"fold{fold}_seed{self.seed}")
            print("Creating the subsets of the dataset")
            train_subset = Subset(dataset, train_idx)
            test_subset = Subset(dataset, val_idx)

            # ===========Weighted datasampler=================
            if self.args.weighted_sampler:
                y = torch.tensor([y for _, y in train_subset]).to(torch.int)
                counts = torch.bincount(y)
                labels_weights = 1.0 / counts
                weights = labels_weights[y]
                sampler = WeightedRandomSampler(
                    weights, len(weights), generator=self.gen
                )
            else:
                sampler = None
            # =================================================

            ccf_train = partial(
                custom_collate_dataset,
                use_augmentation=self.args.augment,
                return_ids=True,
                mode=self.args.dataset_mode,
                norm_mean=self.args.norm_mean,
                norm_std=self.args.norm_std,
                img_ch=self.args.img_ch,
            )
            ccf_test = partial(
                custom_collate_dataset,
                use_augmentation=False,
                return_ids=True,
                mode=self.args.dataset_mode,
                norm_mean=self.args.norm_mean,
                norm_std=self.args.norm_std,
                img_ch=self.args.img_ch,
            )

            train_dataloader = DataLoader(
                train_subset,
                batch_size=self.args.batch_size,
                sampler=sampler,
                num_workers=workers_,
                drop_last=False,
                prefetch_factor=10,
                persistent_workers=True,
                generator=self.gen,
                worker_init_fn=seed_worker,
                collate_fn=ccf_train,
            )
            test_dataloader = DataLoader(
                test_subset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=workers_,
                drop_last=True,
                prefetch_factor=10,
                generator=self.gen,
                worker_init_fn=seed_worker,
                collate_fn=ccf_test,
                persistent_workers=True,
            )

            train_net = copy.deepcopy(self.net)
            train_net = self.train_network_(
                train_net, train_dataloader, test_dataloader, fold
            )

            # ================ Final EVal==============================
            metrics, _ = self.process_epoch_(
                net=train_net,
                dataloader=test_dataloader,
                device=self.device,
                epoch=0,
                fold_n=fold,
                phase="eval",
            )
            if self.args.num_classes == 2:
                acc, precision, recall, pred, labels = (
                    metrics["accuracy"],
                    metrics["precision"],
                    metrics["recall"],
                    torch.tensor(metrics["y_pred"]),
                    torch.tensor(metrics["y_true"]),
                )
            else:
                acc, precision, recall, pred, labels = (
                    metrics["accuracy"],
                    metrics["precision_macro"],
                    metrics["recall_macro"],
                    torch.tensor(metrics["y_pred"]),
                    torch.tensor(metrics["y_true"]),
                )

            # ==========================================================
            folds_precisions.append(precision)
            folds_recalls.append(recall)
            f1_scores.append(
                f1_score(y_true=labels.cpu(), y_pred=pred.cpu(),
                average="binary" if self.args.num_classes == 2 else "macro"),
            )

            folds_accuracies.append(acc)
            if self.args.save_check:
                torch.save(
                    {
                        "weights": train_net.state_dict(),
                        "train_idx": train_idx,
                        "val_idx": val_idx,
                    },
                    f"{self.args.checkpoint_dir}/{self.args.project_name}_{self.args.run_name}_fold{fold}_seed{self.seed}",
                )
            if self.args.wandb_log:
                wandb.finish(0)

        folds_accuracies = torch.tensor(folds_accuracies)
        folds_precisions = torch.tensor(folds_precisions)
        folds_recalls = torch.tensor(folds_recalls)
        f1_scores = torch.tensor(f1_scores)

        # Compute mean and std for all metrics
        metrics_summary = {
            "accuracy": {
                "values": folds_accuracies.tolist(),
                "mean": torch.mean(folds_accuracies).item(),
                "std": torch.std(folds_accuracies).item(),
            },
            "precision": {
                "values": folds_precisions.tolist(),
                "mean": torch.mean(folds_precisions).item(),
                "std": torch.std(folds_precisions).item(),
            },
            "recall": {
                "values": folds_recalls.tolist(),
                "mean": torch.mean(folds_recalls).item(),
                "std": torch.std(folds_recalls).item(),
            },
            "f1_score": {
                "values": f1_scores.tolist(),
                "mean": torch.mean(f1_scores).item(),
                "std": torch.std(f1_scores).item(),
            },
        }

        return metrics_summary

    def train_split(self, dataset):
        # ============= Preparing dataset... ==================
        self.dataset = dataset
        workers_ = self.args.num_workers
        if self.args.freeze_backbone:
            self.net.freeze_backbone()
            print("Backbone has been freezed")

        all_indices = range(len(dataset))
        y_values = [dataset.items[i][1] for i in all_indices]  # Extract labels

        train_idx, val_idx = train_test_split(
            all_indices, test_size=0.3, stratify=y_values, random_state=self.seed
        )
        if self.args.wandb_log:
            self.init_wandb(f"single_split_seed{self.seed}")

        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, val_idx)

        folds_accuracies = []
        folds_precisions = []
        folds_recalls = []
        f1_scores = []

        fold = 0
        if self.args.wandb_log:
            self.init_wandb(f"fold{fold}_seed{self.seed}")

        print("Creating the subsets of the dataset")
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, val_idx)
        # ===========Weighted datasampler=================
        if self.args.weighted_sampler:
            y = torch.tensor([y for _, y in train_subset]).to(torch.int)
            counts = torch.bincount(y)
            labels_weights = 1.0 / counts
            weights = labels_weights[y]
            sampler = WeightedRandomSampler(weights, len(weights), generator=self.gen)
        else:
            sampler = None
        # =================================================
        ccf_train = partial(
            custom_collate_dataset,
            use_augmentation=self.args.augment,
            return_ids=True,
            mode=self.args.dataset_mode,
            norm_mean=self.args.norm_mean,
            norm_std=self.args.norm_std,
            img_ch=self.args.img_ch,
        )
        ccf_test = partial(
            custom_collate_dataset,
            use_augmentation=False,
            return_ids=True,
            mode=self.args.dataset_mode,
            norm_mean=self.args.norm_mean,
            norm_std=self.args.norm_std,
            img_ch=self.args.img_ch,
        )
        train_dataloader = DataLoader(
            train_subset,
            batch_size=self.args.batch_size,
            sampler=sampler,
            num_workers=workers_,
            drop_last=False,
            prefetch_factor=10,
            persistent_workers=True,
            generator=self.gen,
            worker_init_fn=seed_worker,
            collate_fn=ccf_train,
        )
        test_dataloader = DataLoader(
            test_subset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=workers_,
            drop_last=True,
            prefetch_factor=10,
            generator=self.gen,
            worker_init_fn=seed_worker,
            collate_fn=ccf_test,
            persistent_workers=True,
        )
        train_net = copy.deepcopy(self.net)
        train_net = self.train_network_(
            train_net, train_dataloader, test_dataloader, fold
        )
        # ================ Final EVal==============================
        metrics, _ = self.process_epoch_(
            net=train_net,
            dataloader=test_dataloader,
            device=self.device,
            epoch=0,
            fold_n=fold,
            phase="eval",
        )
        acc, precision, recall, pred, labels = (
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            torch.tensor(metrics["y_pred"]),
            torch.tensor(metrics["y_true"]),
        )
        # ==========================================================
        folds_precisions.append(precision)
        folds_recalls.append(recall)
        f1_scores.append(f1_score(y_true=labels.cpu(), y_pred=pred.cpu()))
        folds_accuracies.append(acc)
        if self.args.save_check:
            torch.save(
                {
                    "weights": train_net.state_dict(),
                    "train_idx": train_idx,
                    "val_idx": val_idx,
                },
                f"{self.args.checkpoint_dir}/{self.args.project_name}_{self.args.run_name}_fold{fold}_seed{self.seed}",
            )
        if self.args.wandb_log:
            wandb.finish(0)

        folds_accuracies = torch.tensor(folds_accuracies)
        folds_precisions = torch.tensor(folds_precisions)
        folds_recalls = torch.tensor(folds_recalls)
        f1_scores = torch.tensor(f1_scores)

        # Compute mean and std for all metrics
        metrics_summary = {
            "accuracy": {
                "values": folds_accuracies.tolist(),
                "mean": torch.mean(folds_accuracies).item(),
                "std": torch.std(folds_accuracies).item(),
            },
            "precision": {
                "values": folds_precisions.tolist(),
                "mean": torch.mean(folds_precisions).item(),
                "std": torch.std(folds_precisions).item(),
            },
            "recall": {
                "values": folds_recalls.tolist(),
                "mean": torch.mean(folds_recalls).item(),
                "std": torch.std(folds_recalls).item(),
            },
            "f1_score": {
                "values": f1_scores.tolist(),
                "mean": torch.mean(f1_scores).item(),
                "std": torch.std(f1_scores).item(),
            },
        }

        return metrics_summary

    def train_cv_fake(self, val_dataset, test_dataset, fake_dataset):
        # ============= Preparing dataset... ==================
        workers_ = self.args.num_workers
        if self.args.freeze_backbone:
            self.net.freeze_backbone()
            print("Backbone has been freezed")

        folds_accuracies = []
        folds_precisions = []
        folds_recalls = []
        f1_scores = []
        fold = 0
        if self.args.wandb_log:
            self.init_wandb(f"fold{fold}_seed{self.seed}")

        ccf_train = partial(
            custom_collate_fake_dataset,
            use_augmentation=self.args.augment,
            return_ids=True,
            norm_mean=self.args.norm_mean,
            norm_std=self.args.norm_std,
            img_ch=self.args.img_ch,
        )
        ccf_real = partial(
            custom_collate_dataset,
            use_augmentation=False,
            return_ids=True,
            mode=self.args.dataset_mode,
            norm_mean=self.args.norm_mean,
            norm_std=self.args.norm_std,
            img_ch=self.args.img_ch,
        )

        train_dataloader = DataLoader(
            fake_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=workers_,
            drop_last=True,
            prefetch_factor=10,
            persistent_workers=True,
            generator=self.gen,
            worker_init_fn=seed_worker,
            collate_fn=ccf_train,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=workers_,
            drop_last=True,
            prefetch_factor=10,
            generator=self.gen,
            worker_init_fn=seed_worker,
            collate_fn=ccf_real,
            persistent_workers=True,
        )        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=workers_,
            drop_last=True,
            prefetch_factor=10,
            generator=self.gen,
            worker_init_fn=seed_worker,
            collate_fn=ccf_real,
            persistent_workers=True,
        )
        train_net = copy.deepcopy(self.net)
        train_net = self.train_network_with_val(
            train_net, train_dataloader,val_dataloader, test_dataloader, fold_n=fold
        )
        # ================ Final EVal==============================
        metrics, _ = self.process_epoch_(
            net=train_net,
            dataloader=test_dataloader,
            device=self.device,
            epoch=0,
            fold_n=fold,
            phase="eval",
        )
        acc, precision, recall, pred, labels = (
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            torch.tensor(metrics["y_pred"]),
            torch.tensor(metrics["y_true"]),
        )
        # ==========================================================
        folds_precisions.append(precision)
        folds_recalls.append(recall)
        f1_scores.append(f1_score(y_true=labels.cpu(), y_pred=pred.cpu()))
        folds_accuracies.append(acc)
        if self.args.save_check:
            torch.save(
                {
                    "weights": train_net.state_dict(),
                    "train_idx": [],
                    "val_idx": [],
                },
                f"{self.args.checkpoint_dir}/{self.args.project_name}_{self.args.run_name}_fold{fold}_seed{self.seed}",
            )
        if self.args.wandb_log:
            wandb.finish(0)
        folds_accuracies = torch.tensor(folds_accuracies)
        folds_precisions = torch.tensor(folds_precisions)
        folds_recalls = torch.tensor(folds_recalls)
        f1_scores = torch.tensor(f1_scores)

        # Compute mean and std for all metrics
        metrics_summary = {
            "accuracy": {
                "values": folds_accuracies.tolist(),
                "mean": torch.mean(folds_accuracies).item(),
                "std": torch.std(folds_accuracies).item(),
            },
            "precision": {
                "values": folds_precisions.tolist(),
                "mean": torch.mean(folds_precisions).item(),
                "std": torch.std(folds_precisions).item(),
            },
            "recall": {
                "values": folds_recalls.tolist(),
                "mean": torch.mean(folds_recalls).item(),
                "std": torch.std(folds_recalls).item(),
            },
            "f1_score": {
                "values": f1_scores.tolist(),
                "mean": torch.mean(f1_scores).item(),
                "std": torch.std(f1_scores).item(),
            },
        }

        return metrics_summary


class ContrastiveTrainer:
    def __init__(self, net: BackboneWrapper, device, args: Namespace, opt, scheduler):
        self.device = device
        self.args = args
        self.net = DataParallel(net) if self.args.data_parallel else net.to(self.device)
        self.opt = opt
        self.scheduler = scheduler
        self.seed = 333
        self.gen = torch.Generator().manual_seed(self.seed)
        self.set_seed(self.seed)
        self.loss = (
            HiCoLoss(device=self.device, net=self.net, args=self.args)
            if self.args.backbone == "resnet_fpn_hico"
            else USCLloss(
                net=self.net,
                device=self.device,
                temp=0.5,
                use_sup_loss=self.args.use_sup_loss,
                use_unsup_loss=self.args.use_unsup_loss,
                lambda_=self.args.lambda_,
            )
        )
        self.loss = self.loss.to(self.device)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def init_wandb(self):
        assert wandb.api.api_key, "the api key has not been set!\n"
        wandb.login(verify=True)
        wandb.init(
            project=self.args.project_name,
            name=self.args.run_name,
            config=self.args.__dict__,
        )
        wandb.config.update({"seed": self.seed})

    def train_one_epoch_(self, dataloader, epoch):
        pbar = tqdm(total=len(dataloader), desc=f"Epoch-{epoch}")
        self.net.train()
        for x, xi, xj, y in dataloader:
            loss = self.loss(x, xi, xj, y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            pbar.update(1)
            pbar.set_postfix({"Loss": loss.item()})
            current_lr = self.opt.param_groups[0]["lr"]
            wandb.log({"train_loss": loss.item(), "LR": current_lr})
        self.scheduler.step()

    def train_one_step(self, x, xi, xj, y):
        self.net.train()
        loss = self.loss(x, xi, xj, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        current_lr = self.opt.param_groups[0]["lr"]
        wandb.log(
            {
                "train_loss": loss.item(),
                "learning_rate": current_lr,
            }
        )
        return loss, current_lr

    def warmup_projection_head(self, dataloader, warmup_epochs=5):
        """
        Warm up the projection head while keeping the backbone frozen
        """
        print(f"Starting projection head warmup for {warmup_epochs} epochs")

        # Freeze backbone weights
        for param in self.net.backbone.parameters():
            param.requires_grad = False

        # Only train the projection head
        for param in self.net.contrastive_head.parameters():
            param.requires_grad = True

        warmup_optimizer = torch.optim.Adam(
            self.net.contrastive_head.parameters(),
            lr=self.args.warmup_lr if hasattr(self.args, "warmup_lr") else 1e-3,
        )

        for epoch in range(warmup_epochs):
            pbar = tqdm(total=len(dataloader), desc=f"Warmup-{epoch}")
            self.net.train()
            total_loss = 0.0

            for x, xi, xj, y in dataloader:
                # Move data to the appropriate device
                x, xi, xj, y = (
                    x.to(self.device),
                    xi.to(self.device),
                    xj.to(self.device),
                    y.to(self.device),
                )

                loss = self.loss(x, xi, xj, y)
                warmup_optimizer.zero_grad()
                loss.backward()
                warmup_optimizer.step()
                total_loss += loss.item()

                pbar.update(1)
                pbar.set_postfix({"Warmup Loss": loss.item()})
                wandb.log({"warmup_loss": loss.item()})

            avg_loss = total_loss / len(dataloader)
            print(f"Warmup Epoch {epoch}: Average Loss = {avg_loss:.4f}")

        # Unfreeze backbone for regular training
        for param in self.net.backbone.parameters():
            param.requires_grad = True

        print("Projection head warmup completed")

    def train(self, dataset: torch.utils.data.Dataset):
        self.init_wandb()
        cc_partial = partial(
            custom_collate,
            norm_mean=self.args.norm_mean,
            norm_std=self.args.norm_std,
            img_ch=self.args.img_ch,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=True,
            prefetch_factor=8,
            persistent_workers=True,
            collate_fn=cc_partial,
        )

        # ============ Warmup ========================
        if self.args.warmup_epochs and self.args.warmup_epochs > 0:
            self.warmup_projection_head(dataloader, self.args.warmup_epochs)
        # ============================================
        if self.args.epochs is not None:
            for epoch in range(self.args.epochs):
                self.train_one_epoch_(dataloader, epoch)
                name = os.path.join(
                    self.args.checkpoint_dir,
                    f"{self.args.run_name}_{self.args.dataset_dims}",
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "net_type": type(self.net),
                        "model_state_dict": self.net.backbone.state_dict(),
                    },
                    name,
                )
                print(f"Model saved in {name}.")
        else:
            total_steps = 0
            epoch = 0
            data_iterator = iter(dataloader)
            pbar = tqdm(total=self.args.steps, desc=f"Training by step")

            while total_steps < self.args.steps:
                # Process batches from a full dataloader iteration
                self.net.train()
                for batch in dataloader:
                    if total_steps >= self.args.steps:
                        break

                    x, xi, xj, y = batch
                    loss, lr = self.train_one_step(x, xi, xj, y)
                    if self.scheduler is not None:
                        self.scheduler.step()
                    total_steps += 1
                    epoch += 1
                    pbar.update(1)
                    pbar.set_postfix({"Loss": loss.item(), "LR": lr})
                    if (
                        total_steps % 1000 == 0 or total_steps == self.args.steps
                    ):  # Adjust save frequency as needed
                        name = os.path.join(
                            self.args.checkpoint_dir, f"{self.args.run_name}"
                        )
                        torch.save(
                            {
                                "step": total_steps,
                                "net_type": type(self.net),
                                "model_state_dict": (
                                    self.net.module.backbone.state_dict()
                                    if self.args.data_parallel
                                    else self.net.backbone.state_dict()
                                ),
                            },
                            name,
                        )
                        print(f"Model saved in {name} after {total_steps} steps.")

        wandb.finish(0)
