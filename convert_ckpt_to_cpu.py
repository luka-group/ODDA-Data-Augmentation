import argparse
import torch
import os

parser = argparse.ArgumentParser()

parser.add_argument("--path", default="", type=str)

args = parser.parse_args()

a = torch.load(args.path)

b = dict([(key, val.to("cpu")) for key,val in a.items()])

torch.save(b, os.path.join(os.path.dirname(args.path), "model_cpu.pt"))

