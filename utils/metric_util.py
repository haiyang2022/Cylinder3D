# -*- coding:utf-8 -*-
# author: Xinge
# @file: metric_util.py

import csv
import json
import os

import numpy as np


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iu(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 2)
    hist = hist[unique_label + 1, :]
    hist = hist[:, unique_label + 1]
    return hist


def fast_hist_ignore(pred, label, num_classes, ignore_label=255):
    pred = np.asarray(pred).reshape(-1).astype(np.int64)
    label = np.asarray(label).reshape(-1).astype(np.int64)
    valid = (label != ignore_label) & (label >= 0) & (label < num_classes)
    valid = valid & (pred >= 0) & (pred < num_classes)
    if valid.sum() == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    bin_count = np.bincount(
        num_classes * label[valid] + pred[valid],
        minlength=num_classes ** 2,
    )
    return bin_count[:num_classes ** 2].reshape(num_classes, num_classes)


def per_class_acc(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diag(hist) / hist.sum(1)


def overall_acc(hist):
    total = hist.sum()
    if total == 0:
        return np.nan
    return np.diag(hist).sum() / total


def summarize_hist(hist, class_names):
    iou = per_class_iu(hist)
    acc = per_class_acc(hist)
    summary = {
        "values_are_percent": True,
        "evaluated_points": int(hist.sum()),
        "mIoU": float(np.nanmean(iou) * 100),
        "mAcc": float(np.nanmean(acc) * 100),
        "OA": float(overall_acc(hist) * 100),
        "per_class": [],
    }
    for class_id, class_name in enumerate(class_names):
        summary["per_class"].append({
            "id": int(class_id),
            "class": str(class_name),
            "iou": float(iou[class_id] * 100) if not np.isnan(iou[class_id]) else None,
            "accuracy": float(acc[class_id] * 100) if not np.isnan(acc[class_id]) else None,
            "gt_points": int(hist[class_id, :].sum()),
        })
    return summary


def _format_value(value):
    if value is None or np.isnan(value):
        return "nan"
    return f"{value:.4f}"


def write_metric_files(hist, class_names, output_dir, prefix, metadata=None):
    os.makedirs(output_dir, exist_ok=True)
    metadata = metadata or {}
    summary = summarize_hist(hist, class_names)
    summary["metadata"] = metadata

    txt_path = os.path.join(output_dir, f"{prefix}_metrics.txt")
    json_path = os.path.join(output_dir, f"{prefix}_metrics.json")
    csv_path = os.path.join(output_dir, f"per_class_metrics_{prefix}.csv")

    with open(txt_path, "w") as f:
        f.write("ITC-DATASET-2026 Metrics\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
        f.write(f"evaluated_points: {summary['evaluated_points']}\n")
        f.write("\noverall:\n")
        f.write(f"  mIoU: {_format_value(summary['mIoU'])}\n")
        f.write(f"  mAcc: {_format_value(summary['mAcc'])}\n")
        f.write(f"  OA: {_format_value(summary['OA'])}\n")
        f.write("\nper_class:\n")
        f.write("  id  class             IoU       Acc       gt_points\n")
        for item in summary["per_class"]:
            f.write(
                f"  {item['id']:<3} {item['class']:<16} "
                f"{_format_value(item['iou']):>8} "
                f"{_format_value(item['accuracy']):>8} "
                f"{item['gt_points']:>10}\n"
            )

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "class", "iou", "accuracy", "gt_points"])
        writer.writeheader()
        for item in summary["per_class"]:
            writer.writerow(item)

    return summary, txt_path, json_path, csv_path
