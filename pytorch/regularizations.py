from __future__ import print_function
import numpy as np
import torch


def get_regularization(module_list, reg_type, scale = 1e-6):
    if reg_type == "L1":
        return L1_regularization(module_list, scale = scale)
    elif reg_type == "L2":
        return L2_regularization(module_list, scale = scale)
    else:
        raise Exception("reg_type {0} not recognized!".format(reg_type))


def L1_regularization(module_list, scale = 1e-6):
    if not isinstance(module_list, list):
        module_list = [module_list]
    reg = 0
    for module in module_list:
        for param in module.parameters():
            reg += torch.sum(torch.abs(param))
    return reg * scale


def L2_regularization(module_list, scale = 1e-6):
    if not isinstance(module_list, list):
        module_list = [module_list]
    reg = 0
    for module in module_list:
        for param in module.parameters():
            reg += torch.sum(param ** 2)
    return reg * scale


def integer_regularization(var_list, scale = 1e-6):
    if not isinstance(var_list, list):
        var_list = [var_list]
    reg = 0
    for var in var_list:
        reg += torch.sum(torch.abs(var - torch.round(var)))
    return reg * scale

