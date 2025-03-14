import torch
import torch.distributed as dist
import numpy as np
import random
import re

def get_only_chars(line):

    clean_line = ""

    line = line.lower()
    line = line.replace(" 's", " is") 
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.replace("'", "")

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()
def move_to_device(batch, rank = None):
    ans = {}
    if (rank is None):
        device = 'cuda'
    else:
        device = 'cuda:{}'.format(rank)
    for key in batch:
        try:
            ans[key] = batch[key].to(device = device)
        except Exception as e:
            # print(str(e))
            ans[key] = batch[key]
    return ans
def reduce_loss_dict(loss_dict):
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim = 0)
        dist.reduce(all_losses, dst = 0)
        if dist.get_rank() == 0:
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def read_from_txt_file(input_dir, num_classes):
    lines = open(input_dir,'r').readlines()
    Xs,Ys=[],[]
    count = [0]*num_classes
    for line in lines:
        y,x = line.split('\t')
        y = int(y)
        # if count[y] >= int(434*int(data_split)*2):
        #     continue
        count[y] += 1
        x = x[:-1]
        x = get_only_chars(x)
        Xs.append(x)
        Ys.append(y)
    return Xs, Ys