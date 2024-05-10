import gc
from collections import defaultdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

import tinycudann as tcnn


def chunk_batch(func, chunk_size, move_to_cpu, *args, **kwargs):
    B = None
    for arg in args:
        if isinstance(arg, torch.Tensor): #Assuming the first tensor passed in is rays
            B = arg.shape[0] #Gets the shape of the first tensor - usually num_rays
            break    
    tensor_arg = []
    non_tensor_arg = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            tensor_arg.append(arg)
        else:
            non_tensor_arg.append(arg)
    out = defaultdict(list)
    out_type = None    
    for i in range(0, B, chunk_size): #if the last chunk might be less than chunk_size
        chunked_tensors = [tensor_arg[i:i + chunk_size] for tensor_arg in tensor_arg] #Chunks the tensors rays: (chunk_size, 6)
        out_chunk = func(*chunked_tensors, *non_tensor_arg, **kwargs)
        if out_chunk is None:
            continue
        
        #Checking the type of the output
        out_type = type(out_chunk)
        if isinstance(out_chunk, torch.Tensor):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
            chunk_length = len(out_chunk)
            out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
        elif isinstance(out_chunk, dict):
            pass
        else:
            print(f'Return value of func must be in type [torch.Tensor, list, tuple, dict], get {type(out_chunk)}.')
            exit(1)
            
        #Moving every key to cpu
        for k, v in out_chunk.items():
            v = v if torch.is_grad_enabled() else v.detach()
            v = v.cpu() if move_to_cpu else v
            if k == "ray_indices":
                v = v + i
            out[k].append(v)
    
    if out_type is None:
        return

    out = {k: torch.cat(v, dim=0) for k, v in out.items()}
    if out_type is torch.Tensor:
        return out[0]
    elif out_type in [tuple, list]:
        return out_type([out[i] for i in range(chunk_length)])
    elif out_type is dict:
        return out


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))

trunc_exp = _TruncExp.apply


def get_activation(name):
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == 'none':
        return lambda x: x
    elif name.startswith('scale'):
        scale_factor = float(name[5:])
        return lambda x: x.clamp(0., scale_factor) / scale_factor
    elif name.startswith('clamp'):
        clamp_max = float(name[5:])
        return lambda x: x.clamp(0., clamp_max)
    elif name.startswith('mul'):
        mul_factor = float(name[3:])
        return lambda x: x * mul_factor
    elif name == 'lin2srgb':
        return lambda x: torch.where(x > 0.0031308, torch.pow(torch.clamp(x, min=0.0031308), 1.0/2.4)*1.055 - 0.055, 12.92*x).clamp(0., 1.)
    elif name == 'trunc_exp':
        return trunc_exp
    elif name.startswith('+') or name.startswith('-'):
        return lambda x: x + float(name)
    elif name == 'sigmoid':
        return lambda x: torch.sigmoid(x)
    elif name == 'tanh':
        return lambda x: torch.tanh(x)
    elif name == 'exp':
        return lambda x: torch.exp(x)
    else:
        return getattr(F, name)


def dot(x, y):
    return torch.sum(x*y, -1, keepdim=True)


def reflect(x, n):
    return 2 * dot(x, n) * n - x


def scale_anything(dat, inp_scale, tgt_scale):
    if inp_scale is None:
        inp_scale = [dat.min(), dat.max()]
    
    dat = (dat  - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat



def mapping_dist_to_bin_mitsuba(dists, n_bins, exposure_time, c=1, sigma=5):
    #NOTE: ratio measures the total distance travelled by the light photon, after being emitted from the laser pulse and reflected off the object, divided by the exposure time (the time for which the laser pulse is on)
    times = 2 * dists / c #(2* distances between each sample and their respective origin)
    ratio = times / exposure_time #0.01, basically just 2*distances/exposure_time
    
    #NOTE: For each sample's distance from the origin, we center a bin around it that covers += 4*sigma distance from the sample
    #so ratio-4*sigma is the left edge of the bin and we add ranges because we basically want to get this: for each sample, [ratio-4*sigma, ratio+1-4*sigma, ...], where the middle is roughly that sample's ratio
    ranges = torch.arange(0, 8 * sigma, device=dists.device)[None, :].repeat(ratio.shape[0], 1) #(Create a tensor from 0-23 and repeat it to match the shape of dists)
    bin_mapping = (torch.ceil(ratio-4*sigma))[:, None]+ranges #Creating a bin mapping tensor and adding elementwise elements to ranges, (n_samples, 24)
    
    #NOTE: For each bin mapping entry, we query a normal distribution centered around the true ratio (that sample's distance)
    ranges = bin_mapping - ratio[:, None] 
    dist_weights = torch.exp(-ranges**2/(2*sigma**2))-math.exp(-8) #(n_samples, 24)

    dist_weights[(bin_mapping<0) ] = 0
    dist_weights[(bin_mapping>n_bins) ] = 0

    bin_mapping = torch.clip(bin_mapping, 0, n_bins-1)
    dist_weights = (dist_weights.T/(dist_weights.sum(-1)[: None]+1e-10)).T
    return bin_mapping, dist_weights



def mapping_dist_to_bin(dists, n_bins, exposure_time, c=1):
    times = 2 * dists / c
    #  (torch.randn(times.shape[0])*7).to("cuda")
    ratio = times / exposure_time
    alpha = (torch.ceil(ratio) - ratio) / (torch.ceil(ratio) - torch.floor(ratio) + 1e-10)

    bin_numbers_floor = torch.floor(ratio)
    bin_numbers_ceil = torch.ceil(ratio)
    # if torch.max(bin_numbers)>bin_length:
    #     print("hello")
    bin_numbers_floor = torch.clip(bin_numbers_floor, 0, n_bins - 1)
    bin_numbers_ceil = torch.clip(bin_numbers_ceil, 0, n_bins - 1)

    return bin_numbers_floor, bin_numbers_ceil, alpha


def torch_laser_kernel(laser, device='cuda'):
    m = torch.nn.Conv1d(1, 1, laser.shape[0], padding=(laser.shape[0] - 1) // 2, padding_mode="zeros", device=device)
    m.weight.requires_grad = False
    m.bias.requires_grad = False
    m.bias *= 0
    m.weight = torch.nn.Parameter(laser[None, None, ...])
    return m


def convolve_colour(color, kernel, n_bins):
    color = color.transpose(1, 2).reshape(-1, n_bins)
    color = kernel(color[:, None, :]).squeeze()
    color = color.reshape(-1, 3, n_bins).transpose(1, 2)
    return color



def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    tcnn.free_temporary_memory()
