# -*- coding:utf-8 -*-

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from builder import data_builder, model_builder
from config.config import load_config_data
from dataloader.pc_dataset import get_SemKITTI_label_name
from utils.metric_util import fast_hist_ignore, write_metric_files


def get_class_names(label_mapping, num_class):
    label_name = get_SemKITTI_label_name(label_mapping)
    return [label_name.get(i, f"class_{i}") for i in range(num_class)]


def load_model_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    return checkpoint


def prediction_path_for_frame(prediction_root, data_root, split, frame_path):
    frame_path = Path(frame_path)
    split_root = Path(data_root) / split
    try:
        rel_path = frame_path.relative_to(split_root)
    except ValueError:
        rel_path = Path(frame_path.name)
    return Path(prediction_root) / rel_path.with_suffix(".label")


def evaluate(args):
    configs = load_config_data(args.config_path)
    dataset_config = configs["dataset_params"]
    model_config = configs["model_params"]
    train_hypers = configs["train_params"]

    if args.dataset_root is not None:
        configs["val_data_loader"]["data_path"] = args.dataset_root

    run_dir = args.run_dir or train_hypers.get("run_dir", "./runs/cylinder3d_itc")
    checkpoint_path = args.checkpoint or os.path.join(run_dir, "checkpoints", "best_val_miou.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    split_loader_config = configs["val_data_loader"].copy()
    split_loader_config["imageset"] = args.split
    split_loader_config["shuffle"] = False
    if args.batch_size is not None:
        split_loader_config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        split_loader_config["num_workers"] = args.num_workers

    grid_size = model_config["output_shape"]
    num_class = model_config["num_class"]
    ignore_label = dataset_config["ignore_label"]
    class_names = get_class_names(dataset_config["label_mapping"], num_class)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_builder.build(model_config)
    checkpoint = load_model_checkpoint(model, checkpoint_path)
    model.to(device)
    model.eval()

    dataset_loader, point_dataset = data_builder.build_split(
        dataset_config,
        split_loader_config,
        grid_size=grid_size,
        return_test=True,
    )

    prediction_root = os.path.join(run_dir, "predictions", args.split)
    metrics_dir = os.path.join(run_dir, "metrics")
    hist = np.zeros((num_class, num_class), dtype=np.int64)
    frames = 0

    with torch.no_grad():
        for _, _, val_grid, val_pt_labs, val_pt_fea, indices in tqdm(
                dataset_loader, desc=f"evaluate {args.split}"):
            val_pt_fea_ten = [
                torch.from_numpy(i).float().to(device) for i in val_pt_fea
            ]
            val_grid_ten = [torch.from_numpy(i).to(device) for i in val_grid]
            batch_size = len(val_grid)
            outputs = model(val_pt_fea_ten, val_grid_ten, batch_size)
            voxel_pred = torch.argmax(outputs, dim=1).cpu().numpy()

            for count, grid in enumerate(val_grid):
                point_pred = voxel_pred[count, grid[:, 0], grid[:, 1], grid[:, 2]].astype(np.uint8)
                labels = val_pt_labs[count]
                hist += fast_hist_ignore(point_pred, labels, num_class, ignore_label=ignore_label)

                if not args.no_save_predictions:
                    frame_index = indices[count]
                    frame_path = point_dataset.im_idx[frame_index]
                    pred_path = prediction_path_for_frame(
                        prediction_root,
                        split_loader_config["data_path"],
                        args.split,
                        frame_path,
                    )
                    pred_path.parent.mkdir(parents=True, exist_ok=True)
                    point_pred.tofile(pred_path)
                frames += 1

    sequences = len({Path(path).parent.name for path in point_dataset.im_idx})
    metadata = {
        "method": train_hypers.get("method_name", "Cylinder3D"),
        "run_dir": run_dir,
        "checkpoint": checkpoint_path,
        "dataset_root": split_loader_config["data_path"],
        "split": args.split,
        "sequences": sequences,
        "frames": frames,
        "seed": checkpoint.get("seed", train_hypers.get("seed", "unknown")) if isinstance(checkpoint, dict) else "unknown",
    }
    summary, txt_path, json_path, csv_path = write_metric_files(
        hist,
        class_names,
        metrics_dir,
        args.split,
        metadata=metadata,
    )

    print(f"{args.split} metrics written to: {txt_path}")
    print(f"{args.split} metrics json: {json_path}")
    print(f"{args.split} per-class csv: {csv_path}")
    print("mIoU={:.4f} mAcc={:.4f} OA={:.4f}".format(
        summary["mIoU"],
        summary["mAcc"],
        summary["OA"],
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Cylinder3D on ITC-DATASET-2026")
    parser.add_argument("-y", "--config_path", default="config/itc.yaml")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-save-predictions", action="store_true")
    args = parser.parse_args()

    print(" ".join(sys.argv))
    print(args)
    evaluate(args)
