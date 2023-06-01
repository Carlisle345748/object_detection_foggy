import csv
import json
import os.path

from detectron2.utils.file_io import PathManager


def parse_matric_file(metric_file, val_key):
    metric_train = []
    metric_val = []
    with open(metric_file) as f:
        for line in f:
            data = json.loads(line)
            if val_key in data:
                metric_val.append(data)
            else:
                metric_train.append(data)
    return metric_train, metric_val


def write_cvs(data, output_filename):
    keys = data[0].keys()
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)


def parse_metric_to_csv(result_path, depth=False):
    model_name = os.path.basename(result_path)
    output_path = os.path.join("/Users/carlisle/Developer/cs231n/output/final/metrics_csv", model_name)
    os.makedirs(output_path, exist_ok=True)

    if not PathManager.isdir(result_path):
        raise Exception("invalid path")
    metric_filename = os.path.join(result_path, "metrics.json")
    if not PathManager.isfile(metric_filename):
        raise Exception("metric file not exist")
    val_key = "Depth Delta1" if depth else "cityscapes_fine_instance_seg_val/bbox/AP"
    train, val = parse_matric_file(metric_filename, val_key)
    write_cvs(train, os.path.join(output_path, "train.csv"))
    write_cvs(val, os.path.join(output_path, "val.csv"))


if __name__ == "__main__":
    models_path = "/Users/carlisle/Developer/cs231n/output/final/models/focal"
    for model in PathManager.ls(models_path):
        if model == ".DS_Store":
            continue
        model_path = os.path.join(models_path, model)
        parse_metric_to_csv(model_path, depth=False)


