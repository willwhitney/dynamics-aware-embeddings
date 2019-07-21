import torch
import numpy as np
import scipy.misc
import functools
import os
import math
import matplotlib.pyplot as plt
import imageio
import sys
import pdb
# import ipdb
import shutil
# import git
import copy
import json

from torch.autograd import Variable

def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__")))

def prod(l):
    return functools.reduce(lambda x, y: x * y, l)

def image_tensor(inputs, padding=1):
    # assert is_sequence(inputs)
    assert len(inputs) > 0
    # print(inputs)

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images)-1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding :
                   (i+1) * x_dim + i * padding, :].copy_(image)

        return result

    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x
                  for x in inputs]
        # print(images)
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images)-1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding :
                   (i+1) * y_dim + i * padding].copy_(image)
        return result

def make_image(tensor):
    tensor = tensor.cpu().clamp(0, 1)
    if tensor.size(0) == 1:
        tensor = tensor.expand(3, tensor.size(1), tensor.size(2))
    # import ipdb; ipdb.set_trace()
    return scipy.misc.toimage(tensor.numpy(),
                              high=255*tensor.max().item(),
                              channel_axis=0)

def save_image(filename, tensor):
    img = make_image(tensor)
    img.save(filename)

def save_tensors_image(filename, inputs, padding=1):
    images = image_tensor(inputs, padding)
    return save_image(filename, images)

def save_gif(filename, inputs, bounce=False, color_last=False, duration=0.1):
    images = []
    for tensor in inputs:
        tensor = tensor.cpu()
        if not color_last:
            tensor = tensor.transpose(0,1).transpose(1,2)
        tensor = tensor.clamp(0,1)
        images.append((tensor.cpu().numpy() * 255).astype('uint8'))
    if bounce:
        images = images + list(reversed(images[1:-1]))
    imageio.mimsave(filename, images)


def show(img_tensor):
    if img_tensor.dim() > 2:
        img_tensor = img_tensor.transpose(0, 1).transpose(1, 2)
    # f = plt.figure()
    # plt.imshow(output_tensor.numpy())
    # plt.show()
    # plt.close(f)
    img_tensor = img_tensor.squeeze()
    max_size = 12
    max_input_size = max(img_tensor.size(0), img_tensor.size(1))
    figsize = (torch.Tensor((img_tensor.size(1), img_tensor.size(0)))
               * max_size / max_input_size).ceil()

    fig = plt.figure(figsize=list(figsize))
    if img_tensor.dim() == 2:
        plt.gray()

    plt.imshow(img_tensor.numpy(), interpolation='bilinear')
    plt.show()
    plt.close(fig)


def clip_grad_norm(parameters, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.
    """
    parameters = list(parameters)
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef >= 1:
        return
    for p in parameters:
        p.grad.data.mul_(clip_coef)

def sample(p):
    (mu, sigma) = p
    # noise = mu.clone()
    # noise.normal_(0, 1)
    noise = torch.normal(torch.zeros(mu.size()), torch.ones(sigma.size()))
    noise = noise.type_as(mu.data)
    if isinstance(mu, torch.autograd.variable.Variable):
        noise = torch.autograd.variable.Variable(noise)
    return mu + sigma * noise

def sample_log2(p):
    (mu, log_sigma2) = p
    # noise = mu.clone()
    # noise.normal_(0, 1)
    sigma = torch.exp(0.5 * log_sigma2)

    noise = torch.normal(torch.zeros(mu.size()), torch.ones(sigma.size()))
    noise = noise.type_as(mu.data)
    if isinstance(mu, torch.autograd.variable.Variable):
        noise = torch.autograd.variable.Variable(noise)
    return mu + sigma * noise

def batch_flatten(x):
    return x.resize(x.size(0), prod(x.size()[1:]))

def get_gpu():
    gpu_key = 'CUDA_VISIBLE_DEVICES'
    if gpu_key in os.environ:
        return os.environ[gpu_key]
    else:
        return 'default'


class Histogram(object):
    """
    Ascii histogram
    """
    def __init__(self, data, bins=10):
        """
        Class constructor

        :Parameters:
            - `data`: array like object
        """
        self.data = data
        self.bins = bins
        self.h = np.histogram(self.data, bins=self.bins)
    def horizontal(self, height=4, character ='|'):
        """Returns a multiline string containing a
        a horizontal histogram representation of self.data
        :Parameters:
            - `height`: Height of the histogram in characters
            - `character`: Character to use
        >>> d = normal(size=1000)
        >>> h = Histogram(d,bins=25)
        >>> print h.horizontal(5,'|')
        106            |||
                      |||||
                      |||||||
                    ||||||||||
                   |||||||||||||
        -3.42                         3.09
        """
        his = """"""
        bars = self.h[0]/max(self.h[0])*height
        for l in reversed(range(1,height+1)):
            line = ""
            if l == height:
                line = '%s '%max(self.h[0]) #histogram top count
            else:
                line = ' '*(len(str(max(self.h[0])))+1) #add leading spaces
            for c in bars:
                if c >= math.ceil(l):
                    line += character
                else:
                    line += ' '
            line +='\n'
            his += line
        his += '%.2f'%self.h[1][0] + ' '*(self.bins) +'%.2f'%self.h[1][-1] + '\n'
        return his
    def vertical(self,height=20, character ='|'):
        """
        Returns a Multi-line string containing a
        a vertical histogram representation of self.data
        :Parameters:
            - `height`: Height of the histogram in characters
            - `character`: Character to use
        >>> d = normal(size=1000)
        >>> Histogram(d,bins=10)
        >>> print h.vertical(15,'*')
                              236
        -3.42:
        -2.78:
        -2.14: ***
        -1.51: *********
        -0.87: *************
        -0.23: ***************
        0.41 : ***********
        1.04 : ********
        1.68 : *
        2.32 :
        """
        his = """"""
        xl = ['%.2f'%n for n in self.h[1]]
        lxl = [len(l) for l in xl]
        bars = self.h[0]/max(self.h[0])*height
        his += ' '*int(max(bars)+2+max(lxl))+'%s\n'%max(self.h[0])
        for i,c in enumerate(bars):
            line = xl[i] +' '*int(max(lxl)-lxl[i])+': '+ character*c+'\n'
            his += line
        return his

def clear_progressbar():
    # moves up 2 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")

def grad_norm(model):
    all_grads = [p.grad.data.view(p.grad.data.nelement())
                 for p in model.parameters()
                 if p.grad is not None]
    cat_grads = torch.cat(all_grads, 0)
    return cat_grads.norm()

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# def jupyter_show(tensor):
    # plt.figure()
    # return plt.imshow(tensor.numpy())

def setattrs(obj, attr_dict, exceptions=[]):
    for key in attr_dict:
        if key not in exceptions:
            setattr(obj, key, attr_dict[key])

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    Taken from http://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def atomic_save(obj, path):
    tmp_path = path + '.tmp'
    torch.save(obj, tmp_path)
    shutil.move(tmp_path, path)

def sequence_input(seq, dtype):
    return [Variable(x.type(dtype)) for x in seq]

def conv_out_dim(in_planes, out_planes, in_height, in_width, kernel_size,
                 stride=1, padding=0, dilation=1):
    dilated_kernel = dilation * (kernel_size - 1)
    out_height = math.floor(
        (in_height + 2 * padding - dilated_kernel - 1) / stride + 1)
    out_width = math.floor(
        (in_width + 2 * padding - dilated_kernel - 1) / stride + 1)
    return out_planes, out_height, out_width

def conv_in_dim(out_height, out_width, kernel_size,
                 stride=1, padding=0, dilation=1):
    dilated_kernel = dilation * (kernel_size - 1)
    # (out_height - 1) * stride = in_height + 2 * padding - dilated_kernel - 1
    in_height = math.ceil(
        (out_height - 1) * stride - 2 * padding + dilated_kernel + 1)
    in_width = math.ceil(
        (out_width - 1) * stride - 2 * padding + dilated_kernel + 1)
    return in_height, in_width

def conv_transpose_in_dim(out_height, out_width, kernel_size,
                          stride=1, padding=0, dilation=1):
    # dilated_kernel = dilation * (kernel_size - 1)
    dilated_kernel = kernel_size
    in_height = math.ceil(
        (out_height - dilated_kernel + 2 * padding) / stride + 1)
    in_width = math.ceil(
        (out_width - dilated_kernel + 2 * padding) / stride + 1)
    return in_height, in_width

def conv_transpose_out_dim(in_height, in_width, kernel_size,
                          stride=1, padding=0, dilation=1):
    # dilated_kernel = dilation * (kernel_size - 1)
    dilated_kernel = kernel_size
    out_height = math.ceil(
        (in_height - 1) * stride - 2 * padding + dilated_kernel)
    out_width = math.floor(
        (in_width - 1) * stride - 2 * padding + dilated_kernel)
    return out_height, out_width

def ensure_path_exists(fn):
    """
    A decorator which, given a function that has a path as its first argument,
    ensures that the directory containing that path exists,
    creating it if necessary.
    """
    def wrapper(path, *args, **kwargs):
        try:
            return fn(path, *args, **kwargs)
        except FileNotFoundError:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return fn(path, *args, **kwargs)
    return wrapper

def mean_of_means(tensor_list):
    if type(tensor_list[0]) == torch.autograd.variable.Variable:
        tensor_list = [t.data for t in tensor_list]
    return sum([t.mean() for t in tensor_list]) / len(tensor_list)

def flat_str(x):
    x = x.cpu().detach().view([-1]).numpy()
    fmt_string = "{:+06.3f}\t" * len(x)
    return fmt_string.format(*x)


def serialize_opt(opt):
    # import ipdb; ipdb.set_trace()
    cleaned_opt = copy.deepcopy(vars(opt))
    return json.dumps(cleaned_opt, indent=4, sort_keys=True)

def write_options(opt, location):
    with open(location + "/opt.json", 'w') as f:
        serial_opt = serialize_opt(opt)
        print(serial_opt)
        f.write(serial_opt)
        f.flush()
