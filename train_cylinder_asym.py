# -*- coding:utf-8 -*-

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data
from dataloader.pc_dataset import get_SemKITTI_label_name
from utils.load_save_util import load_checkpoint
from utils.metric_util import fast_hist_ignore, write_metric_files

import warnings

warnings.filterwarnings("ignore")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_class_names(label_mapping, num_class):
    label_name = get_SemKITTI_label_name(label_mapping)
    return [label_name.get(i, f"class_{i}") for i in range(num_class)]


def load_model_if_needed(model, model_load_path):
    if not model_load_path:
        return model
    if not os.path.exists(model_load_path):
        print(f"Model load path does not exist, training from scratch: {model_load_path}")
        return model

    checkpoint = torch.load(model_load_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"Loaded model_state_dict from checkpoint: {model_load_path}")
        return model

    return load_checkpoint(model_load_path, model)


def save_checkpoint(path, model, optimizer, epoch, best_val_miou, best_epoch, config_path,
                    class_names, seed, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_miou": float(best_val_miou),
        "best_epoch": int(best_epoch),
        "config_path": config_path,
        "class_names": class_names,
        "seed": int(seed),
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def evaluate_loader(model, dataset_loader, device, num_class, ignore_label, loss_func,
                    lovasz_softmax):
    model.eval()
    hist = np.zeros((num_class, num_class), dtype=np.int64)
    loss_list = []

    with torch.no_grad():
        for _, val_vox_label, val_grid, val_pt_labs, val_pt_fea in tqdm(
                dataset_loader, desc="validate", leave=False):
            val_pt_fea_ten = [
                torch.from_numpy(i).float().to(device) for i in val_pt_fea
            ]
            val_grid_ten = [torch.from_numpy(i).to(device) for i in val_grid]
            val_label_tensor = val_vox_label.long().to(device)
            val_batch_size = val_vox_label.shape[0]

            predict_labels = model(val_pt_fea_ten, val_grid_ten, val_batch_size)
            loss = lovasz_softmax(
                F.softmax(predict_labels, dim=1),
                val_label_tensor,
                ignore=ignore_label,
            ) + loss_func(predict_labels, val_label_tensor)
            loss_list.append(float(loss.detach().cpu().item()))

            predict_labels = torch.argmax(predict_labels, dim=1).cpu().numpy()
            for count, grid in enumerate(val_grid):
                point_pred = predict_labels[count, grid[:, 0], grid[:, 1], grid[:, 2]]
                hist += fast_hist_ignore(
                    point_pred,
                    val_pt_labs[count],
                    num_class,
                    ignore_label=ignore_label,
                )

    mean_loss = float(np.mean(loss_list)) if loss_list else float("nan")
    return hist, mean_loss


def train(args):
    config_path = args.config_path
    configs = load_config_data(config_path)

    dataset_config = configs["dataset_params"]
    train_dataloader_config = configs["train_data_loader"]
    val_dataloader_config = configs["val_data_loader"]
    model_config = configs["model_params"]
    train_hypers = configs["train_params"]

    if args.dataset_root is not None:
        train_dataloader_config["data_path"] = args.dataset_root
        val_dataloader_config["data_path"] = args.dataset_root

    seed = args.seed if args.seed is not None else train_hypers.get("seed", 0)
    set_random_seed(seed)

    run_dir = args.run_dir or train_hypers.get("run_dir", "./runs/cylinder3d_itc")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    metrics_dir = os.path.join(run_dir, "metrics")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    shutil.copy2(config_path, os.path.join(run_dir, "config_snapshot.yaml"))

    best_checkpoint_path = os.path.join(checkpoint_dir, "best_val_miou.pt")
    latest_checkpoint_path = os.path.join(checkpoint_dir, "latest.pt")

    grid_size = model_config["output_shape"]
    num_class = model_config["num_class"]
    ignore_label = dataset_config["ignore_label"]
    class_names = get_class_names(dataset_config["label_mapping"], num_class)

    max_num_epochs = args.max_epochs if args.max_epochs is not None else train_hypers["max_num_epochs"]
    min_num_epochs = args.min_epochs if args.min_epochs is not None else train_hypers.get("min_num_epochs", 0)
    patience = args.patience if args.patience is not None else train_hypers.get("early_stop_patience", max_num_epochs)
    eval_every_n_epochs = train_hypers.get("eval_every_n_epochs", 1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Run directory: {run_dir}")
    print(f"Seed: {seed}")

    my_model = model_builder.build(model_config)
    my_model = load_model_if_needed(my_model, train_hypers.get("model_load_path", ""))
    my_model.to(device)

    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])
    loss_func, lovasz_softmax = loss_builder.build(
        wce=True,
        lovasz=True,
        num_class=num_class,
        ignore_label=ignore_label,
    )

    train_dataset_loader, val_dataset_loader = data_builder.build(
        dataset_config,
        train_dataloader_config,
        val_dataloader_config,
        grid_size=grid_size,
    )

    best_val_miou = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    history = []
    exit_reason = "max_epochs"

    for epoch in range(max_num_epochs):
        epoch_id = epoch + 1
        my_model.train()
        loss_list = []
        pbar = tqdm(total=len(train_dataset_loader), desc=f"train epoch {epoch_id}")

        for _, train_vox_label, train_grid, _, train_pt_fea in train_dataset_loader:
            train_pt_fea_ten = [
                torch.from_numpy(i).float().to(device) for i in train_pt_fea
            ]
            train_vox_ten = [torch.from_numpy(i).to(device) for i in train_grid]
            label_tensor = train_vox_label.long().to(device)
            train_batch_size = train_vox_label.shape[0]

            optimizer.zero_grad()
            outputs = my_model(train_pt_fea_ten, train_vox_ten, train_batch_size)
            loss = lovasz_softmax(
                F.softmax(outputs, dim=1),
                label_tensor,
                ignore=ignore_label,
            ) + loss_func(outputs, label_tensor)
            loss.backward()
            optimizer.step()

            loss_list.append(float(loss.item()))
            pbar.update(1)

        pbar.close()
        train_loss = float(np.mean(loss_list)) if loss_list else float("nan")

        should_validate = (epoch_id % eval_every_n_epochs == 0) or (epoch_id == max_num_epochs)
        if should_validate:
            hist, val_loss = evaluate_loader(
                my_model,
                val_dataset_loader,
                device,
                num_class,
                ignore_label,
                loss_func,
                lovasz_softmax,
            )
            metadata = {
                "method": train_hypers.get("method_name", "Cylinder3D"),
                "run_dir": run_dir,
                "checkpoint": latest_checkpoint_path,
                "dataset_root": val_dataloader_config["data_path"],
                "split": val_dataloader_config["imageset"],
                "epoch": epoch_id,
                "seed": seed,
            }
            val_summary, val_txt_path, _, _ = write_metric_files(
                hist,
                class_names,
                metrics_dir,
                "val",
                metadata=metadata,
            )
            val_miou = val_summary["mIoU"]
            improved = val_miou > best_val_miou

            if improved:
                best_val_miou = val_miou
                best_epoch = epoch_id
                epochs_without_improvement = 0
                save_checkpoint(
                    best_checkpoint_path,
                    my_model,
                    optimizer,
                    epoch_id,
                    best_val_miou,
                    best_epoch,
                    config_path,
                    class_names,
                    seed,
                )
            else:
                epochs_without_improvement += 1

            history.append({
                "epoch": epoch_id,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mIoU": val_miou,
                "best_val_mIoU": best_val_miou,
                "best_epoch": best_epoch,
            })
            print(
                "epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                "val_mIoU={val_miou:.4f} best_val_mIoU={best:.4f} best_epoch={best_epoch}".format(
                    epoch=epoch_id,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_miou=val_miou,
                    best=best_val_miou,
                    best_epoch=best_epoch,
                )
            )
            print(f"Validation metrics written to: {val_txt_path}")

            if epoch_id >= min_num_epochs and epochs_without_improvement >= patience:
                exit_reason = f"early_stop_patience_{patience}"
                save_checkpoint(
                    latest_checkpoint_path,
                    my_model,
                    optimizer,
                    epoch_id,
                    best_val_miou,
                    best_epoch,
                    config_path,
                    class_names,
                    seed,
                    extra={"exit_reason": exit_reason},
                )
                break
        else:
            history.append({
                "epoch": epoch_id,
                "train_loss": train_loss,
                "val_loss": None,
                "val_mIoU": None,
                "best_val_mIoU": best_val_miou,
                "best_epoch": best_epoch,
            })

        save_checkpoint(
            latest_checkpoint_path,
            my_model,
            optimizer,
            epoch_id,
            best_val_miou,
            best_epoch,
            config_path,
            class_names,
            seed,
        )

    summary = {
        "method": train_hypers.get("method_name", "Cylinder3D"),
        "run_dir": run_dir,
        "config_path": config_path,
        "seed": seed,
        "best_checkpoint": best_checkpoint_path,
        "latest_checkpoint": latest_checkpoint_path,
        "best_epoch": best_epoch,
        "best_val_mIoU": best_val_miou,
        "exit_reason": exit_reason,
        "history": history,
    }
    with open(os.path.join(run_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Training finished: {exit_reason}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Latest checkpoint: {latest_checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cylinder3D on ITC-DATASET-2026")
    parser.add_argument("-y", "--config_path", default="config/itc.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--min-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    args = parser.parse_args()

    print(" ".join(sys.argv))
    print(args)
    train(args)
