
# coding: utf-8

# In[1]:


import numpy as np
import pickle
from itertools import chain
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from AI_scientist.prepare_dataset import Dataset_Gen
from AI_scientist.util import plot_matrices
from AI_scientist.pytorch.net import Net
from AI_scientist.pytorch.util_pytorch import get_activation, get_optimizer, get_criterion, Loss_Fun


# In[2]:


# Definitions:
class Master_Model(nn.Module):
    def __init__(self, statistics_Net = None, generative_Net = None, generative_Net_logstd = None):
        super(Master_Model, self).__init__()
        self.statistics_Net = statistics_Net
        self.generative_Net = generative_Net
        self.generative_Net_logstd = generative_Net_logstd
        self.use_net = "generative"

    @property
    def model_dict(self):
        model_dict = {"type": "Master_Model"}
        model_dict["statistics_Net"] = self.statistics_Net.model_dict
        model_dict["generative_Net"] = self.generative_Net.model_dict
        if self.generative_Net_logstd is not None:
            model_dict["generative_Net_logstd"] = self.generative_Net_logstd.model_dict   
        return model_dict

    def load_model_dict(self, model_dict):
        new_net = load_model_dict(model_dict)
        self.__dict__.update(new_net.__dict__)

    def get_statistics(self, X, y):
        statistics = self.statistics_Net(torch.cat([X, y], 1))
        if isinstance(statistics, tuple):
            statistics = statistics[0]
        else:
            statistics = statistics
        self.generative_Net.set_latent_param(statistics)

    def use_clone_net(self, clone_parameters = True):
        self.cloned_net = clone_net(self.generative_Net, clone_parameters = clone_parameters)
        self.use_net = "cloned"

    def use_generative_net(self):
        self.use_net = "generative"

    def forward(self, X):
        if self.use_net == "generative":
            return self.generative_Net(X)
        elif self.use_net == "cloned":
            return self.cloned_net(X)
        else:
            raise Exception("use_net {0} not recognized!".format(self.use_net))

    def get_regularization(self, source = ["weight", "bias"], target = ["statistics_Net", "generative_Net"], mode = "L1"):
        if target == "all":
            if self.use_net == "generative":
                target = ["statistics_Net", "generative_Net"]
            elif self.use_net == "cloned":
                target = ["cloned_Net"]
            else:
                raise
        if not isinstance(target, list):
            target = [target]
        reg = Variable(torch.FloatTensor(np.array([0])), requires_grad = False)
        for target_ele in target:
            if target_ele == "statistics_Net":
                assert self.use_net == "generative"
                reg = reg + self.statistics_Net.get_regularization(source = source, mode = mode)
            elif target_ele == "generative_Net":
                assert self.use_net == "generative"
                reg = reg + self.generative_Net.get_regularization(source = source, mode = mode)
            elif target_ele == "cloned_Net":
                assert self.use_net == "cloned"
                reg = reg + self.cloned_net.get_regularization(source = source, mode = mode)
            else:
                raise Exception("target element {0} not recognized!".format(target_ele))
        return reg

    def latent_param_quick_learn(self, X, y, validation_data, epochs = 10, batch_size = 128, lr = 1e-2, optim_type = "LBFGS"):
        return self.generative_Net.latent_param_quick_learn(X = X, y = y, validation_data = validation_data, 
                                                            epochs = epochs, batch_size = batch_size, lr = lr, optim_type = optim_type)

    def clone_net_quick_learn(self, X, y, validation_data, batch_size = 128, epochs = 20, lr = 1e-3, optim_type = "adam"):
        self.clone_net_optimizer = get_optimizer(optim_type = optim_type, lr = lr, parameters = self.cloned_net.parameters())
        self.criterion = get_criterion("huber")
        loss_list = []
        X_test, y_test = validation_data
        batch_size = min(batch_size, len(X))
        if isinstance(X, Variable):
            X = X.data
        if isinstance(y, Variable):
            y = y.data

        dataset_train = data_utils.TensorDataset(X, y)
        train_loader = data_utils.DataLoader(dataset_train, batch_size = batch_size, shuffle = True)
        
        for i in range(epochs):
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                X_batch = Variable(X_batch)
                y_batch = Variable(y_batch)
                if optim_type == "LBFGS":
                    def closure():
                        self.clone_net_optimizer.zero_grad()
                        y_pred = self(X_batch)
                        loss = self.criterion(y_pred, y_batch)
                        loss.backward()
                        return loss
                    self.clone_net_optimizer.step(closure)
                else:
                    self.clone_net_optimizer.zero_grad()
                    y_pred = self(X_batch)
                    loss = self.criterion(y_pred, y_batch)
                    loss.backward()
                    self.clone_net_optimizer.step()
            y_pred_test = self(X_test)
            loss = get_criterion("mse")(y_pred_test, y_test)
            loss_list.append(loss.data[0])
        loss_list = np.array(loss_list)
        return loss_list

    
def load_model_dict(model_dict, is_cuda = False):
    if model_dict["type"] == "Statistics_Net":
        net = Statistics_Net(input_size = model_dict["input_size"],
                             pre_pooling_neurons = model_dict["pre_pooling_neurons"],
                             struct_param_pre = model_dict["struct_param_pre"],
                             struct_param_post = model_dict["struct_param_post"],
                             struct_param_post_logvar = model_dict["struct_param_post_logvar"],
                             pooling = model_dict["pooling"],
                             settings = model_dict["settings"],
                             layer_type = model_dict["layer_type"],
                             is_cuda = is_cuda,
                            )
        net.encoding_statistics_Net.load_model_dict(model_dict["encoding_statistics_Net"])
        net.post_pooling_Net.load_model_dict(model_dict["post_pooling_Net"])
        if model_dict["struct_param_post_logvar"] is not None:
            net.post_pooling_logvar_Net.load_model_dict(model_dict["post_pooling_logvar_Net"])
    elif model_dict["type"] == "Generative_Net":
        learnable_latent_param = model_dict["learnable_latent_param"] if "learnable_latent_param" in model_dict else False
        net = Generative_Net(input_size = model_dict["input_size"],
                             W_struct_param_list = model_dict["W_struct_param_list"],
                             b_struct_param_list = model_dict["b_struct_param_list"],
                             num_context_neurons = model_dict["num_context_neurons"],
                             settings_generative = model_dict["settings_generative"],
                             settings_model = model_dict["settings_model"],
                             learnable_latent_param = True,
                             is_cuda = is_cuda,
                            )
        for i, W_struct_param in enumerate(model_dict["W_struct_param_list"]):
            getattr(net, "W_gen_{0}".format(i)).load_model_dict(model_dict["W_gen_{0}".format(i)])
            getattr(net, "b_gen_{0}".format(i)).load_model_dict(model_dict["b_gen_{0}".format(i)])
        if "latent_param" in model_dict and model_dict["latent_param"] is not None:
            if net.latent_param is not None:
                net.latent_param.data.copy_(torch.FloatTensor(model_dict["latent_param"]))
            else:
                net.latent_param = Variable(torch.FloatTensor(model_dict["latent_param"]), requires_grad = False)
        if "context" in model_dict:
            net.context.data.copy_(torch.FloatTensor(model_dict["context"]))
    elif model_dict["type"] in ["Master_Model", "Full_Net"]: # The "Full_Net" name is for legacy
        statistics_Net = load_model_dict(model_dict["statistics_Net"], is_cuda = is_cuda)
        generative_Net = load_model_dict(model_dict["generative_Net"], is_cuda = is_cuda)
        if "generative_Net_logstd" in model_dict:
            generative_Net_logstd = load_model_dict(model_dict["generative_Net_logstd"], is_cuda = is_cuda)
        else:
            generative_Net_logstd = None
        net = Master_Model(statistics_Net = statistics_Net, generative_Net = generative_Net, generative_Net_logstd = generative_Net_logstd)
    else:
        raise Exception("type {0} not recognized!".format(model_dict["type"]))
    return net


class Statistics_Net(nn.Module):
    def __init__(self, input_size, pre_pooling_neurons, struct_param_pre, struct_param_post, struct_param_post_logvar = None, pooling = "max", settings = {"activation": "leakyRelu"}, layer_type = "Simple_layer", is_cuda = False):
        super(Statistics_Net, self).__init__()
        self.input_size = input_size
        self.pre_pooling_neurons = pre_pooling_neurons
        self.struct_param_pre = struct_param_pre
        self.struct_param_post = struct_param_post
        self.struct_param_post_logvar = struct_param_post_logvar
        self.pooling = pooling
        self.settings = settings
        self.layer_type = layer_type
        self.is_cuda = is_cuda

        self.encoding_statistics_Net = Net(input_size = self.input_size, struct_param = self.struct_param_pre, settings = self.settings, is_cuda = is_cuda)
        self.post_pooling_Net = Net(input_size = self.pre_pooling_neurons, struct_param = self.struct_param_post, settings = self.settings, is_cuda = is_cuda)
        if self.struct_param_post_logvar is not None:
            self.post_pooling_logvar_Net = Net(input_size = self.pre_pooling_neurons, struct_param = self.struct_param_post_logvar, settings = self.settings, is_cuda = is_cuda)
        if self.is_cuda:
            self.cuda()

    @property
    def model_dict(self):
        model_dict = {"type": "Statistics_Net"}
        model_dict["input_size"] = self.input_size
        model_dict["pre_pooling_neurons"] = self.pre_pooling_neurons
        model_dict["struct_param_pre"] = self.struct_param_pre
        model_dict["struct_param_post"] = self.struct_param_post
        model_dict["struct_param_post_logvar"] = self.struct_param_post_logvar
        model_dict["pooling"] = self.pooling
        model_dict["settings"] = self.settings
        model_dict["layer_type"] = self.layer_type
        model_dict["encoding_statistics_Net"] = self.encoding_statistics_Net.model_dict
        model_dict["post_pooling_Net"] = self.post_pooling_Net.model_dict
        if self.struct_param_post_logvar is not None:
            model_dict["post_pooling_logvar_Net"] = self.post_pooling_logvar_Net.model_dict
        return model_dict
    
    def load_model_dict(self, model_dict):
        new_net = load_model_dict(model_dict, is_cuda = self.is_cuda)
        self.__dict__.update(new_net.__dict__)

    def forward(self, input):
        encoding = self.encoding_statistics_Net(input)
        if self.pooling == "mean":
            pooled = encoding.mean(0)
        elif self.pooling == "max":
            pooled = encoding.max(0)[0]
        else:
            raise Exception("pooling {0} not recognized!".format(self.pooling))
        output = self.post_pooling_Net(pooled.unsqueeze(0))
        if self.struct_param_post_logvar is None:
            return output
        else:
            logvar = self.post_pooling_logvar_Net(pooled.unsqueeze(0))
            return output, logvar
    
    def forward_inputs(self, X, y):
        return self(torch.cat([X, y], 1))
    

    def get_regularization(self, source = ["weight", "bias"], mode = "L1"):
        reg = self.encoding_statistics_Net.get_regularization(source = source, mode = mode) +               self.post_pooling_Net.get_regularization(source = source, mode = mode)
        if self.struct_param_post_logvar is not None:
            reg = reg + self.post_pooling_logvar_Net.get_regularization(source = source, mode = mode)
        return reg


class Generative_Net(nn.Module):
    def __init__(
        self, 
        input_size,
        W_struct_param_list,
        b_struct_param_list, 
        num_context_neurons = 0, 
        settings_generative = {"activation": "leakyRelu"}, 
        settings_model = {"activation": "leakyRelu"}, 
        learnable_latent_param = False,
        last_layer_linear = False,
        is_cuda = False,
        ):
        super(Generative_Net, self).__init__()
        assert(len(W_struct_param_list) == len(b_struct_param_list))
        self.input_size = input_size
        self.W_struct_param_list = W_struct_param_list
        self.b_struct_param_list = b_struct_param_list
        self.num_context_neurons = num_context_neurons
        self.settings_generative = settings_generative
        self.settings_model = settings_model
        self.learnable_latent_param = learnable_latent_param
        self.last_layer_linear = last_layer_linear
        self.is_cuda = is_cuda

        for i, W_struct_param in enumerate(self.W_struct_param_list):
            setattr(self, "W_gen_{0}".format(i), Net(input_size = self.input_size + num_context_neurons, struct_param = W_struct_param, settings = self.settings_generative))
            setattr(self, "b_gen_{0}".format(i), Net(input_size = self.input_size + num_context_neurons, struct_param = self.b_struct_param_list[i], settings = self.settings_generative))
        # Setting up latent param and context param:
        self.latent_param = nn.Parameter(torch.randn(1, self.input_size)) if learnable_latent_param else None
        if self.num_context_neurons > 0:
            self.context = nn.Parameter(torch.randn(1, self.num_context_neurons))
        if self.is_cuda:
            self.cuda()

    @property
    def model_dict(self):
        model_dict = {"type": "Generative_Net"}
        model_dict["input_size"] = self.input_size
        model_dict["W_struct_param_list"] = self.W_struct_param_list
        model_dict["b_struct_param_list"] = self.b_struct_param_list
        model_dict["num_context_neurons"] = self.num_context_neurons
        model_dict["settings_generative"] = self.settings_generative
        model_dict["settings_model"] = self.settings_model
        model_dict["learnable_latent_param"] = self.learnable_latent_param
        model_dict["last_layer_linear"] = self.last_layer_linear
        for i, W_struct_param in enumerate(self.W_struct_param_list):
            model_dict["W_gen_{0}".format(i)] = getattr(self, "W_gen_{0}".format(i)).model_dict
            model_dict["b_gen_{0}".format(i)] = getattr(self, "b_gen_{0}".format(i)).model_dict
        model_dict["latent_param"] = self.latent_param.data.numpy() if self.latent_param is not None else None
        if hasattr(self, "context"):
            model_dict["context"] = self.context.data.numpy() if not self.is_cuda else self.context.cpu().data.numpy()
        return model_dict
    
    def set_latent_param_learnable(self, mode):
        if mode == "on":
            if not self.learnable_latent_param:
                self.learnable_latent_param = True
                if self.latent_param is None:
                    self.latent_param = nn.Parameter(torch.randn(1, self.input_size))
                else:
                    self.latent_param = nn.Parameter(self.latent_param.data)
            else:
                assert isinstance(self.latent_param, nn.Parameter)
        elif mode == "off":
            if self.learnable_latent_param:
                assert isinstance(self.latent_param, nn.Parameter)
                self.learnable_latent_param = False
                self.latent_param = Variable(self.latent_param.data, requires_grad = False)
            else:
                assert isinstance(self.latent_param, Variable) or self.latent_param is None
        else:
            raise

    def load_model_dict(self, model_dict):
        new_net = load_model_dict(model_dict, is_cuda = self.is_cuda)
        self.__dict__.update(new_net.__dict__)

    def init_weights_bias(self, latent_param):
        if self.num_context_neurons > 0:
            latent_param = torch.cat([latent_param, self.context], 1)
        for i in range(len(self.W_struct_param_list)):
            setattr(self, "W_{0}".format(i), (getattr(self, "W_gen_{0}".format(i))(latent_param)).squeeze(0))
            setattr(self, "b_{0}".format(i), getattr(self, "b_gen_{0}".format(i))(latent_param))       

    def get_weights_bias(self, W_source = None, b_source = None, isplot = False, latent_param = None):
        if latent_param is not None:
            self.init_weights_bias(latent_param)
        W_list = []
        b_list = []
        if W_source is not None:
            for k in range(len(self.W_struct_param_list)):
                if W_source == "core":
                    W = getattr(self, "W_{0}".format(k)).data.numpy()
                else:
                    raise Exception("W_source '{0}' not recognized!".format(W_source))
                W_list.append(W)
        if b_source is not None:
            for k in range(len(self.b_struct_param_list)):
                if b_source == "core":
                    b = getattr(self, "b_{0}".format(k)).data.numpy()
                else:
                    raise Exception("b_source '{0}' not recognized!".format(b_source))
                b_list.append(b)
        if isplot:
            if W_source is not None:
                print("weight {0}:".format(W_source))
                plot_matrices(W_list)
            if b_source is not None:
                print("bias {0}:".format(b_source))
                plot_matrices(b_list)
        return W_list, b_list

    
    def set_latent_param(self, latent_param):
        assert isinstance(latent_param, Variable), "The latent_param must be a Variable!"
        if self.learnable_latent_param:
            self.latent_param.data.copy_(latent_param.data)
        else:
            self.latent_param = latent_param
    
    
    def latent_param_quick_learn(self, X, y, validation_data, batch_size = 128, epochs = 10, lr = 1e-2, optim_type = "LBFGS"):
        assert self.learnable_latent_param is True, "To quick-learn latent_param, you must set learnable_latent_param as True!"
        self.latent_param_optimizer = get_optimizer(optim_type = optim_type, lr = lr, parameters = [self.latent_param])
        self.criterion = get_criterion("huber")
        loss_list = []
        X_test, y_test = validation_data
        batch_size = min(batch_size, len(X))
        if isinstance(X, Variable):
            X = X.data
        if isinstance(y, Variable):
            y = y.data

        dataset_train = data_utils.TensorDataset(X, y)
        train_loader = data_utils.DataLoader(dataset_train, batch_size = batch_size, shuffle = True)
        for i in range(epochs):
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                X_batch = Variable(X_batch)
                y_batch = Variable(y_batch)
                if optim_type == "LBFGS":
                    def closure():
                        self.latent_param_optimizer.zero_grad()
                        y_pred = self(X_batch)
                        loss = self.criterion(y_pred, y_batch)
                        loss.backward()
                        return loss
                    self.latent_param_optimizer.step(closure)
                else:
                    self.latent_param_optimizer.zero_grad()
                    y_pred = self(X_batch)
                    loss = self.criterion(y_pred, y_batch)
                    loss.backward()
                    self.latent_param_optimizer.step()
            y_pred_test = self(X_test)
            loss = get_criterion("mse")(y_pred_test, y_test)
            loss_list.append(loss.data[0])
        loss_list = np.array(loss_list)
        return loss_list


    def forward(self, input, latent_param = None):
        if latent_param is None:
            latent_param = self.latent_param
        self.init_weights_bias(latent_param)
        output = input
        for i in range(len(self.W_struct_param_list)):
            output = torch.matmul(output, getattr(self, "W_{0}".format(i))) + getattr(self, "b_{0}".format(i))
            if i == len(self.W_struct_param_list) - 1 and hasattr(self, "last_layer_linear") and self.last_layer_linear:
                activation = "linear"
            else:
                activation = self.settings_model["activation"] if "activation" in self.settings_model else "leakyRelu"
            output = get_activation(activation)(output)
        return output


    def get_regularization(self, source = ["weight", "bias"], mode = "L1"):
        reg = Variable(torch.FloatTensor(np.array([0])), requires_grad = False)
        for reg_type in source:
            if reg_type == "weight":
                for i in range(len(self.W_struct_param_list)):
                    if mode == "L1":
                        reg = reg + getattr(self, "W_{0}".format(i)).abs().sum()
                    else:
                        raise
            elif reg_type == "bias":
                for i in range(len(self.W_struct_param_list)):
                    if mode == "L1":
                        reg = reg + getattr(self, "b_{0}".format(i)).abs().sum()
                    else:
                        raise
            elif reg_type == "W_gen":
                for i in range(len(self.W_struct_param_list)):
                    reg = reg + getattr(self, "W_gen_{0}".format(i)).get_regularization(source = source, mode = mode)
            elif reg_type == "b_gen":
                for i in range(len(self.W_struct_param_list)):
                    reg = reg + getattr(self, "b_gen_{0}".format(i)).get_regularization(source = source, mode = mode)
            else:
                raise Exception("source {0} not recognized!".format(reg_type))
        return reg


class VAE_Loss(nn.Module):
    def __init__(self, criterion, prior = "Gaussian", beta = 1):
        super(VAE_Loss, self).__init__()
        self.criterion = criterion
        self.prior = "Gaussian"
        self.beta = beta

    def forward(self, input, target, mu, logvar):
        reconstuction_loss = self.criterion(input, target)
        if self.prior == "Gaussian":
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            raise Exception("prior {0} not recognized!".format(self.prior))
        return reconstuction_loss, KLD * self.beta


def get_relevance(X, y, statistics_Net):
    concat = torch.cat([X, y], 1)
    max_datapoint = statistics_Net.encoding_statistics_Net(concat).max(0)[1].data.numpy()
    unique, counts = np.unique(max_datapoint, return_counts = True)
    relevance = np.zeros(len(X))
    relevance[unique] = counts
    return relevance


def sample_Gaussian(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = Variable(torch.randn(std.size()), requires_grad = False)
    return mu + std * eps


def clone_net(generative_Net, layer_type = "Simple_Layer", clone_parameters = True):
    W_init_list = []
    b_init_list = []
    input_size = generative_Net.W_struct_param_list[0][-1][0][0]
    struct_param = []
    statistics = generative_Net.latent_param
    if clone_parameters and generative_Net.num_context_neurons > 0:
        statistics = torch.cat([statistics, generative_Net.context], 1)
    for i in range(len(generative_Net.W_struct_param_list)):
        num_neurons = generative_Net.b_struct_param_list[i][-1][0]
        layer_struct_param = [num_neurons, layer_type, {}]
        struct_param.append(layer_struct_param)
        if clone_parameters:
            W_init_list.append((getattr(generative_Net, "W_gen_{0}".format(i))(statistics)).squeeze(0).data.numpy())
            b_init_list.append(getattr(generative_Net, "b_gen_{0}".format(i))(statistics).data.numpy()[0])
        else:
            W_init_list.append(None)
            b_init_list.append(None)
    return Net(input_size = input_size, struct_param = struct_param, W_init_list = W_init_list, b_init_list = b_init_list, settings = generative_Net.settings_model)


def get_nets(
    input_size,
    output_size,
    pre_pooling_neurons = 60,
    statistics_output_neurons = 10,
    num_context_neurons = 0,
    struct_param_gen_base = None,
    struct_param_pre = None,
    struct_param_post = None,
    struct_param_post_logvar = None,
    statistics_pooling = "mean",
    activation_statistics = "leakyRelu",
    activation_generative = "leakyRelu",
    activation_model = "leakyRelu",
    learnable_latent_param = False,
    isParallel = False,
    is_VAE = False,
    is_uncertainty_net = False,
    ):
    layer_type = "Simple_Layer"
    struct_param_pre = [
        [60, layer_type, {}],
        [60, layer_type, {}],
        [60, layer_type, {}],
        [pre_pooling_neurons, layer_type, {"activation": "linear"}],
    ] if struct_param_pre is None else struct_param_pre
    struct_param_post = [
        [60, layer_type, {}],
        [60, layer_type, {}],
        [statistics_output_neurons, layer_type, {"activation": "linear"}],
    ] if struct_param_post is None else struct_param_post
    if is_VAE or is_uncertainty_net:
        if struct_param_post_logvar is None:
            struct_param_post_logvar = struct_param_post
    statistics_Net = Statistics_Net(input_size = input_size + output_size,
                                    pre_pooling_neurons = pre_pooling_neurons,
                                    struct_param_pre = struct_param_pre,
                                    struct_param_post = struct_param_post,
                                    struct_param_post_logvar = struct_param_post_logvar,
                                    pooling = statistics_pooling,
                                    settings = {"activation": activation_statistics},
                                   )

    # For Generative_Net:
    struct_param_gen_base = [
        [60, layer_type, {}],
        [60, layer_type, {}],
        [60, layer_type, {}],
    ] if struct_param_gen_base is None else struct_param_gen_base
    struct_param_weight1 = struct_param_gen_base + [[(input_size, 20), layer_type, {"activation": "linear"}]]
    struct_param_weight2 = struct_param_gen_base + [[(20, 20), layer_type, {"activation": "linear"}]]
    struct_param_weight3 = struct_param_gen_base + [[(20, output_size), layer_type, {"activation": "linear"}]]
    struct_param_bias1 = struct_param_gen_base + [[20, layer_type, {"activation": "linear"}]]
    struct_param_bias2 = struct_param_gen_base + [[20, layer_type, {"activation": "linear"}]]
    struct_param_bias3 = struct_param_gen_base + [[output_size, layer_type,  {"activation": "linear"}]]
    generative_Net = Generative_Net(input_size = statistics_output_neurons,
                                    num_context_neurons = num_context_neurons,
                                    W_struct_param_list = [struct_param_weight1, struct_param_weight2, struct_param_weight3],
                                    b_struct_param_list = [struct_param_bias1, struct_param_bias2, struct_param_bias3],
                                    settings_generative = {"activation": activation_generative},
                                    settings_model = {"activation": activation_model},
                                    learnable_latent_param = learnable_latent_param,
                                   )
    if is_uncertainty_net:
        generative_Net_logstd = Generative_Net(input_size = statistics_output_neurons,
                                                num_context_neurons = num_context_neurons,
                                                W_struct_param_list = [struct_param_weight1, struct_param_weight2, struct_param_weight3],
                                                b_struct_param_list = [struct_param_bias1, struct_param_bias2, struct_param_bias3],
                                                settings_generative = {"activation": activation_generative},
                                                settings_model = {"activation": activation_model},
                                                learnable_latent_param = learnable_latent_param,
                                               )
    else:
        generative_Net_logstd = None
    if isParallel:
        print("Using Parallel training.")
        statistics_Net = nn.DataParallel(statistics_Net)
        generative_Net = nn.DataParallel(generative_Net)
        if is_uncertainty_net:
            generative_Net_logstd = nn.DataParallel(generative_Net_logstd)
    return statistics_Net, generative_Net, generative_Net_logstd


def get_tasks(task_id_list, num_train, num_test, task_settings = {}, **kwargs):
    num_tasks = num_train + num_test
    tasks = {}
    if "2Dbouncing" in task_id_list:
        num_examples = task_settings["num_examples"] if "num_examples" in task_settings else 2000
        tasks_candidate = get_bouncing_data(num_examples, **kwargs)
        for i, key in enumerate(tasks_candidate.keys()):
            tasks[key] = tasks_candidate[key]
    else:
        for j in range(num_tasks):
            task_id = np.random.choice(task_id_list)
            num_examples = task_settings["num_examples"] if "num_examples" in task_settings else 2000
            if task_id[:12] == "latent_model":
                task = get_latent_model_data(task_settings["z_settings"], settings = task_settings, num_examples = num_examples)
            elif task_id[:10] == "polynomial":
                order = int(task_id.split("_")[1])
                task = get_polynomial_class(task_settings["z_settings"], order = order, settings = task_settings, num_examples = num_examples)
            elif task_id[:8] == "Legendre":
                order = int(task_id.split("_")[1])
                task = get_Legendre_class(task_settings["z_settings"], order = order, settings = task_settings, num_examples = num_examples)
            elif task_id[:6] == "master":
                task_mode = task_id.split("_")[1]
                task = get_master_function(task_settings["z_settings"], mode = task_mode, settings = task_settings, num_examples = num_examples)
            else:
                task = Dataset_Gen(task_id, settings = {"domain": (-3,3),
                                                        "num_train": 200,
                                                        "num_test": 200,
                                                        "isTorch": True,
                                                       })
            for k in range(num_tasks):
                if "{0}_{1}".format(task_id, k) in tasks:
                    continue
                else:
                    task_key = "{0}_{1}".format(task_id, k)
            tasks[task_key] = task
    task_id_train = np.random.choice(list(tasks.keys()), num_train, replace = False).tolist()
    tasks_train = {key: value for key, value in tasks.items() if key in task_id_train}
    tasks_test = {key: value for key, value in tasks.items() if key not in task_id_train}
    tasks_train = OrderedDict(sorted(tasks_train.items(), key=lambda t: t[0]))
    tasks_test = OrderedDict(sorted(tasks_test.items(), key=lambda t: t[0]))
    return tasks_train, tasks_test


def evaluate(task, statistics_Net, generative_Net, generative_Net_logstd = None, criterion = None, is_VAE = False, is_regulated_net = False):
    (_, (X_test, y_test)), _ = task
    loss_fun = Loss_Fun(core = "mse")
    if is_VAE:
        statistics_mu, statistics_logvar = statistics_Net(torch.cat([X_test, y_test], 1))
        statistics_sampled = sample_Gaussian(statistics_mu, statistics_logvar)
        y_pred_sampled = generative_Net(X_test, statistics_sampled)
        loss_sampled, KLD = criterion(y_pred_sampled, y_test, statistics_mu, statistics_logvar)
        
        y_pred = generative_Net(X_test, statistics_mu)
        loss = criterion.criterion(y_pred, y_test)
        mse = loss_fun(y_pred, y_test)
        return loss.data[0], loss_sampled.data[0], mse.data[0], KLD.data[0]
    else:
        if generative_Net_logstd is None:
            statistics = statistics_Net(torch.cat([X_test, y_test], 1))
            if is_regulated_net:
                statistics = get_regulated_statistics(generative_Net, statistics)
            y_pred = generative_Net(X_test, statistics)
            loss = criterion(y_pred, y_test)
            mse = loss_fun(y_pred, y_test)
        else:
            statistics_mu, statistics_logvar = statistics_Net(torch.cat([X_test, y_test], 1))
            if is_regulated_net:
                statistics_mu = get_regulated_statistics(generative_Net, statistics_mu)
                statistics_logvar = get_regulated_statistics(generative_Net_logstd, statistics_logvar)
            y_pred = generative_Net(X_test, statistics_mu)
            y_pred_logstd = generative_Net_logstd(X_test, statistics_logvar)
            loss = criterion(y_pred, y_test, log_std = y_pred_logstd)
            mse = loss_fun(y_pred, y_test)
        return loss.data[0], loss.data[0], mse.data[0], 0


def get_reg(reg_dict, statistics_Net = None, generative_Net = None):
    reg = Variable(torch.FloatTensor([0]), requires_grad = False)
    for net_name, reg_info in reg_dict.items():
        if net_name == "statistics_Net":
            reg_net = statistics_Net
        elif net_name == "generative_Net":
            reg_net = generative_Net
        if isinstance(reg_net, nn.DataParallel):
            reg_net = reg_net.module
        for reg_type, reg_amp in reg_info.items():
            reg = reg + reg_net.get_regularization(source = [reg_type]) * reg_amp
    return reg


def get_regulated_statistics(generative_Net, statistics):
    assert len(statistics.view(-1)) == len(generative_Net.struct_param) * 2 or len(statistics.view(-1)) == len(generative_Net.struct_param)
    if len(statistics.view(-1)) == len(generative_Net.struct_param) * 2:
        statistics = {i: statistics.view(-1)[2*i: 2*i+2] for i in range(len(generative_Net.struct_param))}
    else:
        statistics = {i: statistics.view(-1)[i: i+1] for i in range(len(generative_Net.struct_param))}
    return statistics


def load_trained_models(filename):
    statistics_Net = torch.load(filename + "statistics_Net.pt")
    generative_Net = torch.load(filename + "generative_Net.pt")
    data_record = pickle.load(open(filename + "data.p", "rb"))
    return statistics_Net, generative_Net, data_record


def plot_task_ensembles(tasks, statistics_Net, generative_Net, is_VAE = False, is_regulated_net = False, title = None, isplot = True):
    import matplotlib.pyplot as plt
    statistics_list = []
    z_list = []
    for task_key, task in tasks.items():
        (_, (X_test, y_test)), info = task
        if is_VAE:
            statistics_mu, statistics_logvar = statistics_Net(torch.cat([X_test, y_test], 1))
            statistics = sample_Gaussian(statistics_mu, statistics_logvar)
        else:
            statistics = statistics_Net(torch.cat([X_test, y_test], 1))
            if isinstance(statistics, tuple):
                statistics = statistics[0]
        statistics_list.append(statistics.data.numpy()[0])
        if is_regulated_net:
            statistics = get_regulated_statistics(generative_Net, statistics)
        y_pred = generative_Net(X_test, statistics)
        z_list.append(info["z"])
        if isplot:
            plt.plot(y_test.data.numpy()[:,0], y_pred.data.numpy()[:,0], ".", markersize = 1, alpha = 0.5)
    if title is not None and isplot:
        plt.title(title)
    if isplot:
        plt.show()
    return np.array(statistics_list), np.array(z_list)


def plot_individual_tasks(tasks, statistics_Net, generative_Net, generative_Net_logstd = None, max_plots = 24, 
                          is_VAE = False, is_regulated_net = False, xlim = (-4, 4), sample_times = None):
    import matplotlib.pyplot as plt
    num_columns = 8
    max_plots = max(num_columns * 3, max_plots)
    num_rows = int(np.ceil(max_plots / num_columns))
    fig = plt.figure(figsize = (25, num_rows * 3.3))
    plt.subplots_adjust(hspace = 0.4)
    statistics_list = []
    if len(tasks) > max_plots:
        chosen_id = np.random.choice(list(tasks.keys()), max_plots, replace = False).tolist()
        chosen_id = sorted(chosen_id)
    else:
        chosen_id = sorted(list(tasks.keys()))
    i = 0
    if xlim is not None:
        X_linspace = Variable(torch.linspace(xlim[0], xlim[1], 200).unsqueeze(1))
    for task_id, task in tasks.items():
        ((X_train, y_train), (X_test, y_test)), info = task
        input_size = X_test.size(1)
        chosen_dim = np.random.choice(range(input_size))
        if is_VAE:
            statistics, statistics_logvar = statistics_Net(torch.cat([X_test, y_test], 1))
        else:
            if generative_Net_logstd is None:
                statistics = statistics_Net(torch.cat([X_test, y_test], 1))
            else:
                statistics, statistics_logvar = statistics_Net(torch.cat([X_test, y_test], 1))
        statistics_list.append(statistics.data.numpy().squeeze())
        if task_id not in chosen_id:
            continue
        
        ax = fig.add_subplot(num_rows, num_columns, i + 1)
        if sample_times is None:
            if input_size == 1:
                if is_regulated_net:
                    statistics = get_regulated_statistics(generative_Net, statistics)
                y_pred = generative_Net(X_linspace, statistics)
                ax.plot(X_linspace.data.numpy()[:, chosen_dim], y_pred.data.numpy().squeeze(), "-r", markersize = 3, label = "pred")
                if generative_Net_logstd is not None:
                    if is_regulated_net:
                        statistics_logvar = get_regulated_statistics(generative_Net_logstd, statistics_logvar)
                    y_pred_std = torch.exp(generative_Net_logstd(X_linspace, statistics_logvar))
                    ax.fill_between(X_linspace.data.numpy()[:, chosen_dim], (y_pred - y_pred_std).data.numpy().squeeze(), (y_pred + y_pred_std).data.numpy().squeeze(), color = "r", alpha = 0.3)
#             else:
#                 if is_regulated_net:
#                     statistics = get_regulated_statistics(generative_Net, statistics)
#                 y_pred = generative_Net(X_linspace, statistics)
        else:
            y_pred_list = []
            for j in range(sample_times):
                statistics_sampled = sample_Gaussian(statistics, statistics_logvar)
                y_pred = generative_Net(X_linspace, statistics)
                y_pred_list.append(y_pred.data.numpy())
            y_pred_list = np.concatenate(y_pred_list, 1)
            y_pred_mean = np.mean(y_pred_list, 1)
            y_pred_std = np.std(y_pred_list, 1)
            print(y_pred_std.mean())
            ax.errorbar(X_linspace.data.numpy()[:, chosen_dim], y_pred_mean, yerr = y_pred_std, fmt="-r", markersize = 3, label = "pred")
        ax.plot(X_test.data.numpy()[:, chosen_dim], y_test.data.numpy().squeeze(), ".", markersize = 3, label = "target")
        
        ax.set_xlabel("x_{0}".format(chosen_dim))
        ax.set_ylabel("y")
        ax.set_title(task_id)
        i += 1
    plt.show()
    return [statistics_list]


def get_corrcoef(x, y):
    import scipy
    corrcoef = np.zeros((y.shape[1], x.shape[1]))
    for i in range(corrcoef.shape[0]):
        for j in range(corrcoef.shape[1]):
            corrcoef[i, j] = scipy.stats.pearsonr(y[:,i], x[:, j])[0]
    return corrcoef


def plot_statistics_vs_z(z_list, statistics_list, mode = "corrcoef", title = None):
    import matplotlib.pyplot as plt
    num_columns = 5
    if isinstance(z_list, list):
        z_list = np.stack(z_list, 0)
    if isinstance(statistics_list, list):
        statistics_list = np.stack(statistics_list, 0)
    if len(z_list.shape) == 1:
        z_list = np.expand_dims(z_list, 1)
    z_size = z_list.shape[1]
    num_rows = int(np.ceil(z_size / num_columns))
    color_list = ["b", "r", "g", "y", "c", "m", "k"]
    fig = plt.figure(figsize = (25, num_rows * 3.2))
    
    for i in range(z_size):
        ax = fig.add_subplot(num_rows, num_columns, i + 1)
        for j in range(statistics_list.shape[1]):
            ax.plot(z_list[:,i], statistics_list[:,j], ".{0}".format(color_list[j]), alpha = 0.6, markersize = 2)
            ax.set_title("statistics vs. z_{0}".format(i))
    plt.show()
    
    # Plot coefficient for linear regression:
    if mode == "corrcoef":
        print("statistics (row) vs. z (column) pearsonr correlation coefficient (abs value):")
        corrcoef = get_corrcoef(z_list, statistics_list)
        plot_matrices([np.abs(corrcoef)], title = title)
        print("statistics correlation matrix:")
        plot_matrices([np.abs(np.corrcoef(statistics_list, rowvar = False))])
        print("pca explained variance ratio:")
        pca = PCA()
        pca.fit(statistics_list)
        print(pca.explained_variance_ratio_)
    else:
        print("statistics (row) vs. z (column) linear regression abs(coeff):")
        from sklearn import linear_model
        reg = linear_model.LinearRegression()
        coeff_list = []
        for i in range(statistics_list.shape[1]):
            reg.fit(z_list, statistics_list[:,i])
            coeff_list.append(reg.coef_)
        coeff_list = np.array(coeff_list)
        plot_matrices([np.abs(coeff_list)])


def plot_data_record(data_record, idx = None, is_VAE = False):
    import matplotlib.pyplot as plt
    source = ["loss", "loss_sampled", "mse"] if is_VAE else ["loss", "mse"]
    fig = plt.figure(figsize = (len(source) * 8, 6))
    for i, key in enumerate(source):
        if "{0}_mean_train".format(key) in data_record:
            ax = fig.add_subplot(1, len(source), i + 1)
            if idx is None:
                ax.semilogy(data_record["iter"], data_record["{0}_mean_train".format(key)], label = '{0}_mean_train'.format(key), c = "b")
                ax.semilogy(data_record["iter"], data_record["{0}_mean_test".format(key)], label = '{0}_mean_test'.format(key), c = "r")
                ax.semilogy(data_record["iter"], data_record["{0}_median_train".format(key)], label = '{0}_median_train'.format(key), c = "b", linestyle = "--")
                ax.semilogy(data_record["iter"], data_record["{0}_median_test".format(key)], label = '{0}_median_test'.format(key), c = "r", linestyle = "--")

                ax.legend()
                ax.set_xlabel("training step")
                ax.set_ylabel(key)
                ax.set_title("{0} vs. training step".format(key))
            else:
                loss_train_list = [data_record[key][task_key][idx] for task_key in data_record["tasks_train"][0].keys()]
                loss_test_list = [data_record[key][task_key][-1] for task_key in data_record["tasks_test"][0].keys()]
                ax.hist(loss_train_list, bins = 20, density = True, alpha = 0.3, color="b")
                ax.hist(loss_test_list, bins = 20, density = True, alpha = 0.3, color="r")
                ax.axvline(x= np.mean(loss_train_list), c = "b", alpha = 0.6, label = "train_mean")
                ax.axvline(x= np.median(loss_train_list), c = "b", linestyle = "--", alpha = 0.6, label = "train_median")
                ax.axvline(x= np.mean(loss_test_list), c = "r", alpha = 0.6, label = "test_mean")
                ax.axvline(x= np.median(loss_test_list), c = "r", linestyle = "--", alpha = 0.6, label = "test_median")
                ax.legend()
                ax.set_title("Histogram for {0}:".format(key))
    plt.show()


# In[3]:


def f(x, z, zdim = 1, num_layers = 1, activation = "tanh"):
    """Generating latent-model data:"""
    A0 = lambda z: np.tanh(z)
    A1 = lambda z: z ** 2 / (1 + z ** 2)
    A2 = lambda z: np.sin(z)
    A3 = lambda z: z
    A4 = lambda z: z ** 2 - z
    input_size = x.shape[1]
    if zdim == 1:
        output = x[:,0:1] * A0(z) + x[:,1:2] * A1(z) + x[:,2:3] * A2(z) + x[:,3:4] * A3(z) + A4(z)
        output = get_activation(activation)(output)
        if num_layers >= 2:
            pass
    return output


def get_latent_model_data(
    z_settings = ["Gaussian", (0, 1)],
    settings = {},
    isTorch = True,
    num_examples = 1000,
    ):
    if z_settings[0] == "Gaussian":
        mu, std = z_settings[1]
        z = np.random.randn() * std + mu
    elif z_settings[0] == "uniform":
        zlim = z_settings[1]
        z = np.random.rand() * (zlim[1] - zlim[0]) + zlim[0]
    else:
        raise Exception("z_settings[0] of {0} not recognized!".format(z_settings[0]))
    num_layers = settings["num_layers"] if "num_layers" in settings else 1
    activation = settings["activation"] if "activation" in settings else "tanh"
    xlim = settings["xlim"] if "xlim" in settings else (-5,5)
    input_size = settings["input_size"] if "input_size" in settings else 5
    test_size = settings["test_size"] if "test_size" in settings else 0.2
    
    X = Variable(torch.rand(num_examples, input_size) * (xlim[1] - xlim[0]) + xlim[0], requires_grad = False)
    y = f(X, z, zdim = settings["zdim"], num_layers = num_layers, activation = activation)
    X_train, X_test, y_train, y_test = train_test_split(X.data.numpy(), y.data.numpy(), test_size = test_size)
    if isTorch:
        X_train = Variable(torch.FloatTensor(X_train), requires_grad = False)
        y_train = Variable(torch.FloatTensor(y_train), requires_grad = False)
        X_test = Variable(torch.FloatTensor(X_test), requires_grad = False)
        y_test = Variable(torch.FloatTensor(y_test), requires_grad = False)
    return ((X_train, y_train), (X_test, y_test)), {"z": z}



def get_polynomial_class(
    z_settings = ["Gaussian", (0, 1)],
    order = 3,
    settings = {},
    isTorch = True,
    num_examples = 1000,
    ):
    if z_settings[0] == "Gaussian":
        mu, std = z_settings[1]
        z = np.random.randn(order + 1) * std + mu
    elif z_settings[0] == "uniform":
        zlim = z_settings[1]
        z = np.random.rand(order + 1) * (zlim[1] - zlim[0]) + zlim[0]
    else:
        raise Exception("z_settings[0] of {0} not recognized!".format(z_settings[0]))
    xlim = settings["xlim"] if "xlim" in settings else (-3,3)
    test_size = settings["test_size"] if "test_size" in settings else 0.2
    X = np.random.rand(num_examples, 1) * (xlim[1] - xlim[0]) + xlim[0]
    y = z[0]
    for i in range(1, order + 1):
        y = y + X ** i * z[i]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    if isTorch:
        X_train = Variable(torch.FloatTensor(X_train), requires_grad = False)
        y_train = Variable(torch.FloatTensor(y_train), requires_grad = False)
        X_test = Variable(torch.FloatTensor(X_test), requires_grad = False)
        y_test = Variable(torch.FloatTensor(y_test), requires_grad = False)
    return ((X_train, y_train), (X_test, y_test)), {"z": z}


def get_Legendre_class(
    z_settings = ["Gaussian", (0, 1)],
    order = 3,
    settings = {},
    isTorch = True,
    num_examples = 1000,
    ):
    if z_settings[0] == "Gaussian":
        mu, std = z_settings[1]
        z = np.random.randn(order + 1) * std + mu
    elif z_settings[0] == "uniform":
        zlim = z_settings[1]
        z = np.random.rand(order + 1) * (zlim[1] - zlim[0]) + zlim[0]
    else:
        raise Exception("z_settings[0] of {0} not recognized!".format(z_settings[0]))
    xlim = settings["xlim"] if "xlim" in settings else (-1,1)
    test_size = settings["test_size"] if "test_size" in settings else 0.2
    
    X = np.random.rand(num_examples, 1) * (xlim[1] - xlim[0]) + xlim[0]
    y = np.polynomial.legendre.legval(X, z)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    if isTorch:
        X_train = Variable(torch.FloatTensor(X_train), requires_grad = False)
        y_train = Variable(torch.FloatTensor(y_train), requires_grad = False)
        X_test = Variable(torch.FloatTensor(X_test), requires_grad = False)
        y_test = Variable(torch.FloatTensor(y_test), requires_grad = False)
    return ((X_train, y_train), (X_test, y_test)), {"z": z}


def get_master_function(
    z_settings = ["Gaussian", (0, 1)],
    mode = "sawtooth",
    settings = {},
    isTorch = True,
    num_examples = 1000,
    ):
    def trianglewave(
        x,
        frequency = 0.25,
        height = 1,
        ):
        remainder = x % (1 / float(frequency))
        slope = height * frequency * 2
        return 2 * np.minimum(slope * remainder, 2 * height - slope * remainder) - 1
    def S(x):
        return np.sin(x * np.pi / 2)
    def Gaussian(x):
        return 1 / np.sqrt(2 * np.pi) * np.exp(- x ** 2 / 2)
    def Softplus(x):
        return np.log(1 + np.exp(x))
    if z_settings[0] == "Gaussian":
        mu, std = z_settings[1]
        z = np.random.randn(4) * std + mu
    elif z_settings[0] == "uniform":
        zlim = z_settings[1]
        z = np.random.rand(4) * (zlim[1] - zlim[0]) + zlim[0]
    else:
        raise Exception("z_settings[0] of {0} not recognized!".format(z_settings[0]))
    xlim = settings["xlim"] if "xlim" in settings else (-3,3)
    test_size = settings["test_size"] if "test_size" in settings else 0.2
    z[0] = np.abs(z[0]) + 0.5
    z[2] = np.abs(z[2] + 1)
    frequency = z[0]
    x0 = z[1]
    amp = z[2]
    const = z[3]
    
    X = np.random.rand(num_examples, 1) * (xlim[1] - xlim[0]) + xlim[0]
    if mode == "sawtooth":
        f = trianglewave
    elif mode == "sin":
        f = S
    elif mode == "tanh":
        f = np.tanh
    elif mode == "Gaussian":
        f = Gaussian
    elif mode == "softplus":
        f = Softplus
    else:
        raise Exception("mode {0} not recognized!".format(mode))
    
    y = f((X - x0) * frequency) * amp + const
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    if isTorch:
        X_train = Variable(torch.FloatTensor(X_train), requires_grad = False)
        y_train = Variable(torch.FloatTensor(y_train), requires_grad = False)
        X_test = Variable(torch.FloatTensor(X_test), requires_grad = False)
        y_test = Variable(torch.FloatTensor(y_test), requires_grad = False)
    return ((X_train, y_train), (X_test, y_test)), {"z": z}


def get_bouncing_data(num_examples, **kwargs):
    num_examples = num_examples * 10
    from AI_scientist.variational.util_variational import get_env_data
    tasks = {}
    i = 0
    for env_name in ["envBounce2"]:
        render = kwargs["render"] if "render" in kwargs else False
        dataset = get_env_data("envBounce2", num_examples = num_examples, isplot = False, is_cuda = False, episode_length = 200, render = render)
        ((X_train, y_train), (X_test, y_test), (reflected_train, reflected_test)), info = dataset
        bouncing_modes = np.unique(reflected_train.data.numpy())
        for bounce_mode in bouncing_modes:
            if bounce_mode == 0:
                continue
            task_name = "{0}-{1}".format(env_name, bounce_mode)
            bounce_mode = float(bounce_mode)
            X_train_task = X_train[reflected_train.unsqueeze(1) == bounce_mode].view(-1, X_train.size(1))
            y_train_task = y_train[reflected_train.unsqueeze(1) == bounce_mode].view(-1, y_train.size(1))
            X_test_task = X_test[reflected_test.unsqueeze(1) == bounce_mode].view(-1, X_test.size(1))
            y_test_task = y_test[reflected_test.unsqueeze(1) == bounce_mode].view(-1, y_test.size(1))
            tasks[task_name] = ((X_train_task, y_train_task), (X_test_task, y_test_task)), {"z": i}
            i += 1
    return tasks

