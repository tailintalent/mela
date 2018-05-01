from __future__ import print_function
import numpy as np
from copy import deepcopy
import random
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import _Loss



class Loss_Fun(nn.Module):
    def __init__(self, core = "mse", epsilon = 1e-10):
        super(Loss_Fun, self).__init__()
        self.name = "Loss_Fun"
        self.core = core
        self.epsilon = epsilon

    def forward(self, pred, target, sample_weights = None, is_mean = True):
        if self.core == "huber":
            loss = nn.SmoothL1Loss(reduce = False)(pred, target)
        elif self.core == "mse":
            loss = get_criterion(self.core, reduce = False)(pred, target)
        elif self.core == "mae":
            loss = get_criterion(self.core, reduce = False)(pred, target)
        elif self.core == "mse-conv":
            loss = get_criterion(self.core, reduce = False)(pred, target).mean(-1).mean(-1)
        elif self.core == "mlse":
            loss = torch.log(nn.MSELoss(reduce = False)(pred, target) + self.epsilon)
        elif self.core == "mse+mlse":
            loss = torch.log(nn.MSELoss(reduce = False)(pred, target) + self.epsilon) + nn.MSELoss(reduce = False)(pred, target).mean()
        else:
            raise Exception("loss mode {0} not recognized!".format(self.core))
        if sample_weights is not None:
            assert tuple(loss.size()) == tuple(sample_weights.size())
            loss = loss * sample_weights
        if is_mean:
            loss = loss.mean()
        return loss


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    sample_size = len(labels_dense)
    labels_one_hot = np.zeros((sample_size, num_classes))
    labels_one_hot[np.arange(sample_size), np.array(labels_dense).astype(int)] = 1
    return labels_one_hot.astype(int)


def get_name(List, max_number = 10000):
    for k in range(max_number):
        if str(k) not in List:
            return str(k)
    raise Exception("Too much items in List")


def record_data(data_record_dict, data_list, key_list, nolist = False):
    """Record data to the dictionary data_record_dict. It records each key: value pair in the corresponding location of 
    key_list and data_list into the dictionary."""
    assert len(data_list) == len(key_list), "the data_list and key_list should have the same length!"
    for data, key in zip(data_list, key_list):
        if nolist:
            data_record_dict[key] = data
        else:
            if key not in data_record_dict:
                data_record_dict[key] = [data]
            else: 
                data_record_dict[key].append(data)


class Batch_Generator(object):
    def __init__(self, X, y, batch_size = 50, target_one_hot_off = False):
        """
        Initilize the Batch_Generator class
        """
        if isinstance(X, tuple):
            self.binary_X = True
        else:
            self.binary_X = False
        if isinstance(X, Variable):
            X = X.data.numpy()
        if isinstance(y, Variable):
            y = y.data.numpy()
        self.target_one_hot_off = target_one_hot_off

        if self.binary_X:
            X1, X2 = X
            self.sample_length = len(y)
            assert len(X1) == len(X2) == self.sample_length, "X and y must have the same length!"
            assert batch_size <= self.sample_length, "batch_size must not be larger than \
                    the number of samples!"
            self.batch_size = int(batch_size)

            X1_len = [element.shape[-1] for element in X1]
            X2_len = [element.shape[-1] for element in X2]
            y_len = [element.shape[-1] for element in y]
            if len(np.unique(X1_len)) == 1 and len(np.unique(X2_len)) == 1:
                assert len(np.unique(y_len)) == 1, \
                    "when X1 and X2 has only one size, the y should also have only one size!"
                self.combine_size = False
                self.X1 = np.array(X1)
                self.X2 = np.array(X2)
                self.y = np.array(y)
                self.index = np.array(range(self.sample_length))
                self.idx_batch = 0
                self.idx_epoch = 0
            else:
                self.combine_size = True
                self.X1 = X1
                self.X2 = X2
                self.y = y
                self.input_dims_list = zip(X1_len, X2_len)
                self.input_dims_dict = {}
                for i, input_dims in enumerate(self.input_dims_list):
                    if input_dims not in self.input_dims_dict:
                        self.input_dims_dict[input_dims] = {"idx":[i]}
                    else:
                        self.input_dims_dict[input_dims]["idx"].append(i)
                for input_dims in self.input_dims_dict:
                    idx = np.array(self.input_dims_dict[input_dims]["idx"])
                    self.input_dims_dict[input_dims]["idx"] = idx
                    self.input_dims_dict[input_dims]["X1"] = [X1[i] for i in idx]
                    self.input_dims_dict[input_dims]["X2"] = [X2[i] for i in idx]
                    self.input_dims_dict[input_dims]["y"] = [y[i] for i in idx]
                    self.input_dims_dict[input_dims]["idx_batch"] = 0
                    self.input_dims_dict[input_dims]["idx_epoch"] = 0
                    self.input_dims_dict[input_dims]["index"] = np.array(range(len(idx))).astype(int)
        else:
            self.sample_length = len(y)
            assert batch_size <= self.sample_length, "batch_size must not be larger than \
                    the number of samples!"
            self.batch_size = int(batch_size)

            X_len = [element.shape[-1] for element in X]
            y_len = [element.shape[-1] for element in y]

            if len(np.unique(X_len)) == 1 and len(np.unique(y_len)) == 1:
                assert len(np.unique(y_len)) == 1, \
                    "when X has only one size, the y should also have only one size!"
                self.combine_size = False
                self.X = np.array(X)
                self.y = np.array(y)
                self.index = np.array(range(self.sample_length))
                self.idx_batch = 0
                self.idx_epoch = 0
            else:
                self.combine_size = True
                self.X = X
                self.y = y
                self.input_dims_list = zip(X_len, y_len)
                self.input_dims_dict = {}
                for i, input_dims in enumerate(self.input_dims_list):
                    if input_dims not in self.input_dims_dict:
                        self.input_dims_dict[input_dims] = {"idx":[i]}
                    else:
                        self.input_dims_dict[input_dims]["idx"].append(i)
                for input_dims in self.input_dims_dict:
                    idx = np.array(self.input_dims_dict[input_dims]["idx"])
                    self.input_dims_dict[input_dims]["idx"] = idx
                    self.input_dims_dict[input_dims]["X"] = [X[i] for i in idx]
                    self.input_dims_dict[input_dims]["y"] = [y[i] for i in idx]
                    self.input_dims_dict[input_dims]["idx_batch"] = 0
                    self.input_dims_dict[input_dims]["idx_epoch"] = 0
                    self.input_dims_dict[input_dims]["index"] = np.array(range(len(idx))).astype(int)


    def reset(self):
        """Reset the index and batch iteration to the initialization state"""
        if not self.combine_size:
            self.index = np.array(range(self.sample_length))
            self.idx_batch = 0
            self.idx_epoch = 0
        else:
            for input_dims in self.input_dims_dict:
                self.input_dims_dict[input_dims]["idx_batch"] = 0
                self.input_dims_dict[input_dims]["idx_epoch"] = 0
                self.input_dims_dict[input_dims]["index"] = np.array(range(len(idx))).astype(int)


    def next_batch(self, mode = "random", isTorch = False, is_cuda = False, given_dims = None):
        """Generate each batch with the same size (even if the examples and target may have variable size)"""
        if self.binary_X:
            if not self.combine_size:
                start = self.idx_batch * self.batch_size
                end = (self.idx_batch + 1) * self.batch_size

                if end > self.sample_length:
                    self.idx_epoch += 1
                    self.idx_batch = 0
                    np.random.shuffle(self.index)
                    start = 0
                    end = self.batch_size

                self.idx_batch += 1
                chosen_index = self.index[start:end]
                y_batch = deepcopy(self.y[chosen_index])
                if self.target_one_hot_off:
                    y_batch = y_batch.argmax(-1)
                X1_batch = deepcopy(self.X1[chosen_index])
                X2_batch = deepcopy(self.X2[chosen_index])
            else: # If the input_dims have variable size
                if mode == "random":
                    if given_dims is None:
                        rand = np.random.choice(self.sample_length)
                        input_dims = self.input_dims_list[rand]
                    else:
                        input_dims = given_dims
                    length = len(self.input_dims_dict[input_dims]["idx"])
                    if self.batch_size >= length:
                        chosen_index = np.random.choice(length, size = self.batch_size, replace = True)
                    else:
                        start = self.input_dims_dict[input_dims]["idx_batch"] * self.batch_size
                        end = (self.input_dims_dict[input_dims]["idx_batch"] + 1) * self.batch_size
                        if end > length:
                            self.input_dims_dict[input_dims]["idx_epoch"] += 1
                            self.input_dims_dict[input_dims]["idx_batch"] = 0
                            np.random.shuffle(self.input_dims_dict[input_dims]["index"])
                            start = 0
                            end = self.batch_size
                        self.input_dims_dict[input_dims]["idx_batch"] += 1
                        chosen_index = self.input_dims_dict[input_dims]["index"][start:end]
                    y_batch = deepcopy(np.array([self.input_dims_dict[input_dims]["y"][j] for j in chosen_index]))
                    if self.target_one_hot_off:
                        y_batch = y_batch.argmax(-1)
                    X1_batch = deepcopy(np.array([self.input_dims_dict[input_dims]["X1"][j] for j in chosen_index]))
                    X2_batch = deepcopy(np.array([self.input_dims_dict[input_dims]["X2"][j] for j in chosen_index]))

            if isTorch:
                X1_batch = Variable(torch.from_numpy(X1_batch), requires_grad = False).type(torch.FloatTensor)
                X2_batch = Variable(torch.from_numpy(X2_batch), requires_grad = False).type(torch.FloatTensor)
                if self.target_one_hot_off:
                    y_batch = Variable(torch.from_numpy(y_batch), requires_grad = False).type(torch.LongTensor)
                else:
                    y_batch = Variable(torch.from_numpy(y_batch), requires_grad = False).type(torch.FloatTensor)

                if is_cuda:
                    X1_batch = X1_batch.cuda()
                    X2_batch = X2_batch.cuda()
                    y_batch = y_batch.cuda()

            return (X1_batch, X2_batch), y_batch
        else:
            if not self.combine_size:
                start = self.idx_batch * self.batch_size
                end = (self.idx_batch + 1) * self.batch_size

                if end > self.sample_length:
                    self.idx_epoch += 1
                    self.idx_batch = 0
                    np.random.shuffle(self.index)
                    start = 0
                    end = self.batch_size

                self.idx_batch += 1
                chosen_index = self.index[start:end]
                y_batch = deepcopy(self.y[chosen_index])
                if self.target_one_hot_off:
                    y_batch = y_batch.argmax(-1)
                X_batch = deepcopy(self.X[chosen_index])
            else: # If the input_dims have variable size
                if mode == "random":
                    if given_dims is None:
                        rand = np.random.choice(self.sample_length)
                        input_dims = self.input_dims_list[rand]
                    else:
                        input_dims = given_dims
                    length = len(self.input_dims_dict[input_dims]["idx"])
                    if self.batch_size >= length:
                        chosen_index = np.random.choice(length, size = self.batch_size, replace = True)
                    else:
                        start = self.input_dims_dict[input_dims]["idx_batch"] * self.batch_size
                        end = (self.input_dims_dict[input_dims]["idx_batch"] + 1) * self.batch_size
                        if end > length:
                            self.input_dims_dict[input_dims]["idx_epoch"] += 1
                            self.input_dims_dict[input_dims]["idx_batch"] = 0
                            np.random.shuffle(self.input_dims_dict[input_dims]["index"])
                            start = 0
                            end = self.batch_size
                        self.input_dims_dict[input_dims]["idx_batch"] += 1
                        chosen_index = self.input_dims_dict[input_dims]["index"][start:end]
                    y_batch = deepcopy(np.array([self.input_dims_dict[input_dims]["y"][j] for j in chosen_index]))
                    if self.target_one_hot_off:
                        y_batch = y_batch.argmax(-1)
                    X_batch = deepcopy(np.array([self.input_dims_dict[input_dims]["X"][j] for j in chosen_index]))
            if isTorch:
                X_batch = Variable(torch.from_numpy(X_batch), requires_grad = False).type(torch.FloatTensor)
                if self.target_one_hot_off:
                    y_batch = Variable(torch.from_numpy(y_batch), requires_grad = False).type(torch.LongTensor)
                else:
                    y_batch = Variable(torch.from_numpy(y_batch), requires_grad = False).type(torch.FloatTensor)

                if is_cuda:
                    X_batch = X_batch.cuda()
                    y_batch = y_batch.cuda()

            return X_batch, y_batch



class Gradient_Noise_Scale_Gen(object):
    def __init__(
        self,
        gamma = 0.55,
        eta = 0.01,
        noise_scale_start = 1e-2,
        noise_scale_end = 1e-6,
        gradient_noise_interval_batch = 10,
        fun_pointer = "generate_scale_simple",
        batch_size = 50,
        start_epoch = 0,
        ):
        self.gamma = gamma
        self.eta = eta
        self.noise_scale_start = noise_scale_start
        self.noise_scale_end = noise_scale_end
        self.gradient_noise_interval_batch = gradient_noise_interval_batch
        self.batch_size = batch_size
        self.start_epoch = 0
        self.generate_scale = getattr(self, fun_pointer) # Sets the default function to generate scale
        
    def get_max_iter(self, epochs, num_examples):
        self.epochs = epochs
        self.num_examples = num_examples
        self.max_iter = int((self.epochs + 1 - self.start_epoch) * self.num_examples / self.batch_size / self.gradient_noise_interval_batch) + 1
    
    def generate_scale_simple(
        self,
        epochs,
        num_examples,
        verbose = True
        ):
        self.get_max_iter(epochs, num_examples)       
        gradient_noise_scale = np.sqrt(self.eta * (np.array(range(self.max_iter)) + 1) ** (- self.gamma))
        if verbose:
            print("gradient_noise_scale: start = {0}, end = {1:.6f}, gamma = {2}, length = {3}".format(gradient_noise_scale[0], gradient_noise_scale[-1], self.gamma, self.max_iter))
        return gradient_noise_scale

    def generate_scale_fix_ends(
        self,
        epochs,
        num_examples,
        verbose = True,
        ):
        self.get_max_iter(epochs, num_examples)
        ratio = (self.noise_scale_start / float(self.noise_scale_end)) ** (1 / self.gamma) - 1
        self.bb = self.max_iter / ratio
        self.aa = self.noise_scale_start * self.bb ** self.gamma
        gradient_noise_scale = np.sqrt(self.aa * (np.array(range(self.max_iter)) + self.bb) ** (- self.gamma))
        if verbose:
            print("gradient_noise_scale: start = {0}, end = {1:.6f}, gamma = {2}, length = {3}".format(gradient_noise_scale[0], gradient_noise_scale[-1], self.gamma, self.max_iter))
        return gradient_noise_scale


def visualize_graph(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    from graphviz import Digraph
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d' % v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot


def to_Variable(X, requires_grad = False):
    if isinstance(X, list):
        return [Variable(torch.from_numpy(element), requires_grad = requires_grad).type(torch.FloatTensor) for element in X]
    else:
        return Variable(torch.from_numpy(X), requires_grad = requires_grad).type(torch.FloatTensor)


def softmax(X, axis = -1):
    if isinstance(X, Variable) or isinstance(X, torch.Tensor):
        X_max = torch.max(X, axis, keepdim = True)[0]
        X = torch.exp(X - X_max)
        return X / X.sum(dim = axis, keepdim = True)
    else:
        X_max = np.amax(X, axis, keepdims = True)
        X = np.exp(X - X_max)
        return X / X.sum(axis = axis, keepdims = True)


def get_class_vars(obj):
    import inspect
    return {k:v for k, v in inspect.getmembers(obj)
        if not k.startswith('__') and not inspect.ismethod(v)}


def set_class_vars(obj, Dict):
    for k, v in Dict.items():
        setattr(obj, k, v)


def make_dir(filename):
    import os
    import errno
    if not os.path.exists(os.path.dirname(filename)):
        print("directory {0} does not exist, created.".format(os.path.dirname(filename)))
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                print(exc)
                raise


def pp_list(List, num_digits = 3, level = 0):
    if np.array(List).size == 1:
        List = np.array(List.flatten())
        if level == 0:
            ele = List[0]
        elif level == 1:
            ele = List[0][0]
        elif level == 2:
            ele = List[0][0][0]
        return "{0:.{1}f}".format(ele, num_digits)
    else:
        output = ""
        for element in np.array(List).flatten():
            if output != "":
                output += ", "
            if level == 0:
                ele = element
            elif level == 1:
                ele = element[0]
            elif level == 2:
                ele = element[0][0]
            output += "{0:.{1}f}".format(ele, num_digits)
        output = "[" + output + "]"
        return output


def init_module_weights(module_list, init_weights_mode = "glorot-normal"):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        if init_weights_mode == "glorot-uniform":
            glorot_uniform_limit = np.sqrt(6 / float(module.in_features + module.out_features))
            module.weight.data.uniform_(-glorot_uniform_limit, glorot_uniform_limit)
        elif init_weights_mode == "glorot-normal":
            glorot_normal_std = np.sqrt(2 / float(module.in_features + module.out_features))
            module.weight.data.normal_(mean = 0, std = glorot_normal_std)
        else:
            raise Exception("init_weights_mode '{0}' not recognized!".format(init_weights_mode))


def init_module_bias(module_list, init_bias_mode = "zeros"):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        if init_bias_mode == "zeros":
            module.bias.data.fill_(0)
        else:
            raise Exception("init_bias_mode '{0}' not recognized!".format(init_bias_mode))


def init_weight(weight_list, init):
    if not isinstance(weight_list, list):
        weight_list = [weight_list]
    for weight in weight_list:
        if len(weight.size()) == 2:
            rows = weight.size(0)
            columns = weight.size(1)
        elif len(weight.size()) == 1:
            rows = 1
            columns = weight.size(0)
        if init is None:
            init = "glorot-normal"
        if not isinstance(init, str):
            weight.data.copy_(torch.FloatTensor(init))
        else:
            if init == "glorot-normal":
                glorot_normal_std = np.sqrt(2 / float(rows + columns))
                weight.data.normal_(mean = 0, std = glorot_normal_std)
            else:
                raise Exception("init '{0}' not recognized!".format(init))

def init_bias(bias_list, init):
    if not isinstance(bias_list, list):
        bias_list = [bias_list]
    for bias in bias_list:
        if init is None:
            init = "zeros"
        if not isinstance(init, str):
            bias.data.copy_(torch.FloatTensor(init))
        else:
            if init == "zeros":
                bias.data.fill_(0)
            else:
                raise Exception("init '{0}' not recognized!".format(init))


def get_activation(activation):
    if activation == "linear":
        f = lambda x: x
    elif activation == "relu":
        f = F.relu
    elif activation == "leakyRelu":
        f = nn.LeakyReLU(negative_slope = 0.3)
    elif activation == "tanh":
        f = F.tanh
    elif activation == "softplus":
        f = F.softplus
    elif activation == "sigmoid":
        f = F.sigmoid
    elif activation == "selu":
        f = F.selu
    elif activation == "elu":
        f = F.elu
    elif activation == "sign":
        f = lambda x: torch.sign(x)
    elif activation == "heaviside":
        f = lambda x: (torch.sign(x) + 1) / 2.
    else:
        raise Exception("activation {0} not recognized!".format(activation))
    return f


class MAELoss(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(MAELoss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input, target):
        assert not target.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"
        loss = (input - target).abs()
        if self.reduce:
            if self.size_average:
                loss = loss.mean()
            else:
                loss = loss.sum()
        return loss


def get_criterion(loss_type, reduce = True, **kwargs):
    if loss_type == "huber":
        criterion = nn.SmoothL1Loss(reduce = reduce)
    elif loss_type == "mse":
        criterion = nn.MSELoss(reduce = reduce)
    elif loss_type == "mae":
        criterion = MAELoss(reduce = reduce)
    elif loss_type == "cross-entropy":
        criterion = nn.CrossEntropyLoss(reduce = reduce)
    elif loss_type == "Loss_with_uncertainty":
        criterion = Loss_with_uncertainty(core = kwargs["loss_core"] if "loss_core" in kwargs else "mse", epsilon = 1e-6)
    else:
        raise Exception("loss_type {0} not recognized!".format(loss_type))
    return criterion


def get_optimizer(optim_type, lr, parameters):
    if optim_type == "adam":
        optimizer = optim.Adam(parameters, lr = lr)
    elif optim_type == "RMSprop":
        optimizer = optim.RMSprop(parameters, lr = lr)
    elif optim_type == "LBFGS":
        optimizer = optim.LBFGS(parameters, lr = lr)
    else:
        raise Exception("optim_type {0} not recognized!".format(optim_type))
    return optimizer


class Loss_with_uncertainty(nn.Module):
    def __init__(self, core = "mse", epsilon = 1e-6):
        super(Loss_with_uncertainty, self).__init__()
        self.name = "Loss_with_uncertainty"
        self.core = core
        self.epsilon = epsilon
    
    def forward(self, pred, target, log_std = None, std = None, sample_weights = None, is_mean = True):
        if self.core == "mse":
            loss_core = get_criterion(self.core, reduce = False)(pred, target) / 2
        elif self.core == "mae":
            loss_core = get_criterion(self.core, reduce = False)(pred, target)
        elif self.core == "huber":
            loss_core = get_criterion(self.core, reduce = False)(pred, target)
        elif self.core == "mlse":
            loss_core = torch.log((target - pred) ** 2 + 1e-10)
        elif self.core == "mse+mlse":
            loss_core = (target - pred) ** 2 / 2 + torch.log((target - pred) ** 2 + 1e-10)
        else:
            raise Exception("loss's core {0} not recognized!".format(self.core))
        if std is not None:
            assert log_std is None
            loss = loss_core / (self.epsilon + std ** 2) + torch.log(std + 1e-7)
        else:
            loss = loss_core / (self.epsilon + torch.exp(2 * log_std)) + log_std
        if sample_weights is not None:
            sample_weights = sample_weights.view(loss.size())
            loss = loss * sample_weights
        if is_mean:
            loss = loss.mean()
        return loss


def zero_grad_hook(idx):
    def hook_function(grad):
        grad[idx] = 0
        return grad
    return hook_function


def get_full_struct_param_ele(struct_param, settings):
    struct_param_new = deepcopy(struct_param)
    for i, layer_struct_param in enumerate(struct_param_new):
        if settings is not None and layer_struct_param[1] != "Symbolic_Layer":
            layer_struct_param[2] = {key: value for key, value in deepcopy(settings).items() if key in ["activation"]}
            layer_struct_param[2].update(struct_param[i][2])
        else:
            layer_struct_param[2] = deepcopy(struct_param[i][2])
    return struct_param_new


def get_full_struct_param(struct_param, settings):
    struct_param_new_list = []
    if isinstance(struct_param, tuple):
        for i, struct_param_ele in enumerate(struct_param):
            if isinstance(settings, tuple):
                settings_ele = settings[i]
            else:
                settings_ele = settings
            struct_param_new_list.append(get_full_struct_param_ele(struct_param_ele, settings_ele))
        return tuple(struct_param_new_list)
    else:
        return get_full_struct_param_ele(struct_param, settings)
