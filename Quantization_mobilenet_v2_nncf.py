# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2023/5/26 14:23
# @Author : liumin
# @File : Quantization_mobilenet_v2_nncf.py

import os
import re
import torch
import torch.nn as nn
import torchvision
import subprocess
from pathlib import Path
from typing import List, Optional

import numpy as np
import openvino.runtime as ov
import torch
from openvino.tools import mo
from openvino.tools.pot import save_model
from sklearn.metrics import accuracy_score
from torchvision import datasets
from torchvision import models
from torchvision import transforms
from tqdm import tqdm

import nncf

from SlimPytorch.quantization.mobilenet_v2 import MobileNetV2

# Set the data and model directories
DATA_DIR = '/home/liumin/data/hymenoptera/val'
MODEL_DIR = './weights'



def load_pretrain_model(model_dir):
    model = MobileNetV2('mobilenet_v2', classifier=True)
    num_ftrs = model.fc[1].in_features
    model.fc[1] = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_dir, map_location='cpu'))
    return model

def load_val_data(data_dir):
    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_dataset = datasets.ImageFolder(data_dir, data_transform)
    dataload = torch.utils.data.DataLoader(image_dataset, batch_size=16, shuffle=False, num_workers=4)
    return dataload


model = load_pretrain_model(Path(MODEL_DIR) / 'mobilenet_v2_train.pt')
model.eval()


val_loader = load_val_data(DATA_DIR)


def transform_fn(data_item):
    images, _ = data_item
    return images

calibration_dataset = nncf.Dataset(val_loader, transform_fn)
quantized_model = nncf.quantize(model, calibration_dataset)


ov_model = mo.convert_model(model.cpu(), input_shape=[-1, 3, 224, 224])
ov_quantized_model = mo.convert_model(quantized_model.cpu(), input_shape=[-1, 3, 224, 224])



def get_model_size(ir_path: str, m_type: str = "Mb", verbose: bool = True) -> float:
    xml_size = os.path.getsize(ir_path)
    bin_size = os.path.getsize(os.path.splitext(ir_path)[0] + ".bin")
    for t in ["bytes", "Kb", "Mb"]:
        if m_type == t:
            break
        xml_size /= 1024
        bin_size /= 1024
    model_size = xml_size + bin_size
    if verbose:
        print(f"Model graph (xml):   {xml_size:.3f} Mb")
        print(f"Model weights (bin): {bin_size:.3f} Mb")
        print(f"Model size:          {model_size:.3f} Mb")
    return model_size


def run_benchmark(model_path: str, shape: Optional[List[int]] = None, verbose: bool = True) -> float:
    command = f"benchmark_app -m {model_path} -d CPU -api async -t 15"
    if shape is not None:
        command += f' -shape [{",".join(str(x) for x in shape)}]'
    cmd_output = subprocess.check_output(command, shell=True)  # nosec
    if verbose:
        print(*str(cmd_output).split("\\n")[-9:-1], sep="\n")
    match = re.search(r"Throughput\: (.+?) FPS", str(cmd_output))
    return float(match.group(1))


def validate(model: ov.Model, val_loader: torch.utils.data.DataLoader) -> float:
    predictions = []
    references = []

    compiled_model = ov.compile_model(model)
    output = compiled_model.outputs[0]

    for images, target in tqdm(val_loader):
        pred = compiled_model(images)[output]
        predictions.append(np.argmax(pred, axis=1))
        references.append(target)

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)
    return accuracy_score(predictions, references)


fp32_ir_path = f"weights/mobilenet_v2_fp32.xml"
ov.serialize(ov_model, fp32_ir_path)
print(f"[1/7] Save FP32 model: {fp32_ir_path}")
fp32_model_size = get_model_size(fp32_ir_path, verbose=True)

int8_ir_path = f"weights/mobilenet_v2_int8.xml"
ov.serialize(ov_quantized_model, int8_ir_path)
print(f"[2/7] Save INT8 model: {int8_ir_path}")
int8_model_size = get_model_size(int8_ir_path, verbose=True)

print("[3/7] Benchmark FP32 model:")
# fp32_fps = run_benchmark(fp32_ir_path, shape=[1, 3, 224, 224], verbose=True)
print("[4/7] Benchmark INT8 model:")
# int8_fps = run_benchmark(int8_ir_path, shape=[1, 3, 224, 224], verbose=True)

print("[5/7] Validate OpenVINO FP32 model:")
fp32_top1 = validate(ov_model, val_loader)
print(f"Accuracy @ top1: {fp32_top1:.3f}")

print("[6/7] Validate OpenVINO INT8 model:")
int8_top1 = validate(ov_quantized_model, val_loader)
print(f"Accuracy @ top1: {int8_top1:.3f}")

print("[7/7] Report:")
print(f"Accuracy drop: {fp32_top1 - int8_top1:.3f}")
print(f"Model compression rate: {fp32_model_size / int8_model_size:.3f}")
# https://docs.openvino.ai/latest/openvino_docs_optimization_guide_dldt_optimization_guide.html
print(f"Performance speed up (throughput mode): {int8_fps / fp32_fps:.3f}")