
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
import torch.nn.functional as F
from torch.autograd import Variable
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from mela.prepare_dataset import Dataset_Gen
from mela.util import plot_matrices
from mela.settings.a2c_env_settings import ENV_SETTINGS_CHOICE
from mela.settings.global_param import COLOR_LIST
from mela.pytorch.net import Net, ConvNet
from mela.pytorch.util_pytorch import get_activation, get_optimizer, get_criterion, Loss_Fun, to_Variable, to_np_array, to_one_hot, flatten
from mela.variational.util_variational import sort_datapoints, predict_forward, reshape_time_series


# In[2]:


# Definitions:
class Master_Model(nn.Module):
    def __init__(self, statistics_Net = None, generative_Net = None, generative_Net_logstd = None, is_cuda = False):
        super(Master_Model, self).__init__()
        self.statistics_Net = statistics_Net
        self.generative_Net = generative_Net
        self.generative_Net_logstd = generative_Net_logstd
        self.use_net = "generative"
        self.is_cuda = is_cuda

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
    
    def get_clone_net(self, X = None, y = None, clone_parameters = True):
        if X is not None or y is not None:
            self.get_statistics(X, y)
        return clone_net(self.generative_Net, clone_parameters = clone_parameters)

    def use_generative_net(self):
        self.use_net = "generative"

    def forward(self, X):
        if self.use_net == "generative":
            return self.generative_Net(X)
        elif self.use_net == "cloned":
            return self.cloned_net(X)
        else:
            raise Exception("use_net {0} not recognized!".format(self.use_net))

    
    def get_predictions(
        self,
        X_test,
        X_train,
        y_train,
        is_time_series = True,
        is_VAE = False,
        is_uncertainty_net = False,
        is_regulated_net = False,
        forward_steps = [1],
        ):
        results = {}
        if is_VAE:
            statistics_mu, statistics_logvar = self.statistics_Net.forward_inputs(X_train, y_train)
            statistics = sample_Gaussian(statistics_mu, statistics_logvar)
            results["statistics_mu"] = statistics_mu
            results["statistics_logvar"] = statistics_logvar
            results["statistics"] = statistics
            if is_regulated_net:
                statistics = get_regulated_statistics(generative_Net, statistics)
                results["statistics_feed"] = statistics
            y_pred = self.generative_Net(X_test, statistics)
            results["y_pred"] = y_pred
        else:
            if is_uncertainty_net:
                statistics_mu, statistics_logvar = self.statistics_Net.forward_inputs(X_train, y_train)
                results["statistics_mu"] = statistics_mu
                results["statistics_logvar"] = statistics_logvar
                results["statistics"] = statistics
                if is_regulated_net:
                    statistics_mu = get_regulated_statistics(self.generative_Net, statistics_mu)
                    statistics_logvar = get_regulated_statistics(self.generative_Net_logstd, statistics_logvar)
                    results["statistics_mu_feed"] = statistics_mu
                    results["statistics_logvar_feed"] = statistics_logvar    
                y_pred = self.generative_Net(X_test, statistics_mu)
                y_pred_logstd = self.generative_Net_logstd(X_test, statistics_logvar)
                
                results["y_pred"] = y_pred
                results["y_pred_logstd"] = y_pred_logstd
            else:
                statistics = self.statistics_Net.forward_inputs(X_train, y_train)
                results["statistics"] = statistics
                if is_regulated_net:
                    statistics = get_regulated_statistics(self.generative_Net, statistics)
                    results["statistics_feed"] = statistics
                y_pred = get_forward_pred(self.generative_Net, X_test, forward_steps, is_time_series = is_time_series, latent_param = statistics, jump_step = 2, is_flatten = True)
                results["y_pred"] = y_pred
        return results


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
        if self.is_cuda:
            reg = reg.cuda()
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

    def latent_param_quick_learn(self, X, y, validation_data, loss_core = "huber", epochs = 10, batch_size = 128, lr = 1e-2, optim_type = "LBFGS", reset_latent_param = False):
        if reset_latent_param:
            self.get_statistics(X, y)
        return self.generative_Net.latent_param_quick_learn(X = X, y = y, validation_data = validation_data, loss_core = loss_core,
                                                            epochs = epochs, batch_size = batch_size, lr = lr, optim_type = optim_type)

    def clone_net_quick_learn(self, X, y, validation_data, loss_core = "huber", epochs = 40, batch_size = 128, lr = 1e-3, optim_type = "adam"):
        mse_list, self.cloned_net = quick_learn(self.cloned_net, X, y, validation_data, loss_core = loss_core, batch_size = batch_size, epochs = epochs, lr = lr, optim_type = optim_type)
        return mse_list


def quick_learn(model, X, y, validation_data, forward_steps = [1], is_time_series = True, loss_core = "huber", batch_size = 128, epochs = 40, lr = 1e-3, optim_type = "adam"):
    model_train = deepcopy(model)
    net_optimizer = get_optimizer(optim_type = optim_type, lr = lr, parameters = model_train.parameters())
    criterion = get_criterion(loss_core)
    mse_list = []
    X_test, y_test = validation_data
    batch_size = min(batch_size, len(X))
    if isinstance(X, Variable):
        X = X.data
    if isinstance(y, Variable):
        y = y.data

    dataset_train = data_utils.TensorDataset(X, y)
    train_loader = data_utils.DataLoader(dataset_train, batch_size = batch_size, shuffle = True)

    y_pred_test = get_forward_pred(model_train, X_test, forward_steps = forward_steps, is_time_series = is_time_series, jump_step = 2, is_flatten = True)
    mse_test = get_criterion("mse")(y_pred_test, y_test)
    mse_list.append(mse_test.data[0])
    for i in range(epochs):
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = Variable(X_batch)
            y_batch = Variable(y_batch)
            if optim_type == "LBFGS":
                def closure():
                    net_optimizer.zero_grad()
                    y_pred = get_forward_pred(model_train, X_batch, forward_steps = forward_steps, is_time_series = is_time_series, jump_step = 2, is_flatten = True)
                    loss = criterion(y_pred, y_batch)
                    loss.backward()
                    return loss
                net_optimizer.step(closure)
            else:
                net_optimizer.zero_grad()
                y_pred = get_forward_pred(model_train, X_batch, forward_steps = forward_steps, is_time_series = is_time_series, jump_step = 2, is_flatten = True)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                net_optimizer.step()
        y_pred_test = get_forward_pred(model_train, X_test, forward_steps = forward_steps, is_time_series = is_time_series, jump_step = 2, is_flatten = True)
        mse_test = get_criterion("mse")(y_pred_test, y_test)
        mse_list.append(mse_test.data[0])
    mse_list = np.array(mse_list)
    return mse_list, model_train

    
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
    elif model_dict["type"] == "Statistics_Net_Conv":
        net = Statistics_Net_Conv(input_channels = model_dict["input_channels"],
                                  num_cls = model_dict["num_cls"],
                                  pre_pooling_neurons = model_dict["pre_pooling_neurons"],
                                  struct_param_pre_conv = model_dict["struct_param_pre_conv"],
                                  struct_param_pre = model_dict["struct_param_pre"],
                                  struct_param_post = model_dict["struct_param_post"],
                                  struct_param_post_logvar = model_dict["struct_param_post_logvar"],
                                  pooling = model_dict["pooling"],
                                  settings = model_dict["settings"],
                                  layer_type = model_dict["layer_type"],
                                  is_cuda = is_cuda,
                                 )
        net.encoding_statistics_ConvNet.load_model_dict(model_dict["encoding_statistics_ConvNet"])
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
                if is_cuda:
                    net.latent_param = net.latent_param.cuda()
        if "context" in model_dict:
            net.context.data.copy_(torch.FloatTensor(model_dict["context"]))
    elif model_dict["type"] == "Generative_Net_Conv":
        learnable_latent_param = model_dict["learnable_latent_param"] if "learnable_latent_param" in model_dict else False
        net = Generative_Net_Conv(input_channels = model_dict["input_channels"],
                             latent_size = model_dict["latent_size"],
                             W_struct_param_list = model_dict["W_struct_param_list"],
                             b_struct_param_list = model_dict["b_struct_param_list"],
                             struct_param_model = model_dict["struct_param_model"],
                             num_context_neurons = model_dict["num_context_neurons"],
                             settings_generative = model_dict["settings_generative"],
                             settings_model = model_dict["settings_model"],
                             learnable_latent_param = True,
                             is_cuda = is_cuda,
                            )
        for i in range(len(model_dict["struct_param_model"])):
            if model_dict["struct_param_model"][i][1] in model_dict["param_available"]:
                getattr(net, "W_gen_{0}".format(i)).load_model_dict(model_dict["W_gen_{0}".format(i)])
                getattr(net, "b_gen_{0}".format(i)).load_model_dict(model_dict["b_gen_{0}".format(i)])
        if "latent_param" in model_dict and model_dict["latent_param"] is not None:
            if net.latent_param is not None:
                net.latent_param.data.copy_(torch.FloatTensor(model_dict["latent_param"]))
            else:
                net.latent_param = Variable(torch.FloatTensor(model_dict["latent_param"]), requires_grad = False)
                if is_cuda:
                    net.latent_param = net.latent_param.cuda()
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

    
class Statistics_Net_Conv(nn.Module):
    def __init__(self, input_channels, num_cls, pre_pooling_neurons, struct_param_pre_conv, struct_param_pre, struct_param_post, struct_param_post_logvar = None, pooling = "max", settings = {"activation": "leakyRelu"}, layer_type = "Simple_layer", is_cuda = False):
        super(Statistics_Net_Conv, self).__init__()
        self.input_channels = input_channels
        self.num_cls = num_cls
        self.pre_pooling_neurons = pre_pooling_neurons
        self.struct_param_pre_conv = struct_param_pre_conv
        self.struct_param_pre = struct_param_pre
        self.struct_param_post = struct_param_post
        self.struct_param_post_logvar = struct_param_post_logvar
        self.pooling = pooling
        self.settings = settings
        self.layer_type = layer_type
        self.is_cuda = is_cuda

        self.encoding_statistics_ConvNet = ConvNet(input_channels = self.input_channels, struct_param = self.struct_param_pre_conv, settings = self.settings, is_cuda = is_cuda)
        X = Variable(torch.zeros(10, 3, 28, 28))
        if is_cuda:
            X = X.cuda()
        dim_enc_conv = flatten(self.encoding_statistics_ConvNet(X)[0]).size(1)
        self.encoding_statistics_Net = Net(input_size = dim_enc_conv + num_cls, struct_param = self.struct_param_pre, settings = self.settings, is_cuda = is_cuda)
        self.post_pooling_Net = Net(input_size = self.pre_pooling_neurons, struct_param = self.struct_param_post, settings = self.settings, is_cuda = is_cuda)
        if self.struct_param_post_logvar is not None:
            self.post_pooling_logvar_Net = Net(input_size = self.pre_pooling_neurons, struct_param = self.struct_param_post_logvar, settings = self.settings, is_cuda = is_cuda)
        if self.is_cuda:
            self.cuda()

    def model_dict(self):
        model_dict = {"type": "Statistics_Net_Conv"}
        model_dict["input_channels"] = self.input_channels
        model_dict["num_cls"] = self.num_cls
        model_dict["pre_pooling_neurons"] = self.pre_pooling_neurons
        model_dict["struct_param_pre_conv"] = self.struct_param_pre_conv
        model_dict["struct_param_pre"] = self.struct_param_pre
        model_dict["struct_param_post"] = self.struct_param_post
        model_dict["struct_param_post_logvar"] = self.struct_param_post_logvar
        model_dict["pooling"] = self.pooling
        model_dict["settings"] = self.settings
        model_dict["layer_type"] = self.layer_type
        model_dict["encoding_statistics_Net"] = self.encoding_statistics_Net.model_dict
        model_dict["encoding_statistics_ConvNet"] = self.encoding_statistics_ConvNet.model_dict
        model_dict["post_pooling_Net"] = self.post_pooling_Net.model_dict
        if self.struct_param_post_logvar is not None:
            model_dict["post_pooling_logvar_Net"] = self.post_pooling_logvar_Net.model_dict
        return model_dict

    def load_model_dict(self, model_dict):
        new_net = load_model_dict(model_dict, is_cuda = self.is_cuda)
        self.__dict__.update(new_net.__dict__)

    def forward(self, X, y):
        encoding_X, _ = self.encoding_statistics_ConvNet(X)
        encoding_X = flatten(encoding_X)
        encoding = torch.cat([encoding_X, to_one_hot(y, self.num_cls)], 1)
        encoding = self.encoding_statistics_Net(encoding)
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
        return self(X, y)
    

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
        last_layer_linear = True,
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
            setattr(self, "W_gen_{0}".format(i), Net(input_size = self.input_size + num_context_neurons, struct_param = W_struct_param, settings = self.settings_generative, is_cuda = is_cuda))
            setattr(self, "b_gen_{0}".format(i), Net(input_size = self.input_size + num_context_neurons, struct_param = self.b_struct_param_list[i], settings = self.settings_generative, is_cuda = is_cuda))
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
        if self.latent_param is None:
            model_dict["latent_param"] = None
        else:
            model_dict["latent_param"] = self.latent_param.cpu().data.numpy() if self.is_cuda else self.latent_param.data.numpy()
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
    
    
    def latent_param_quick_learn(self, X, y, validation_data, loss_core = "huber", epochs = 10, batch_size = 128, lr = 1e-2, optim_type = "LBFGS"):
        assert self.learnable_latent_param is True, "To quick-learn latent_param, you must set learnable_latent_param as True!"
        self.latent_param_optimizer = get_optimizer(optim_type = optim_type, lr = lr, parameters = [self.latent_param])
        self.criterion = get_criterion(loss_core)
        loss_list = []
        X_test, y_test = validation_data
        batch_size = min(batch_size, len(X))
        if isinstance(X, Variable):
            X = X.data
        if isinstance(y, Variable):
            y = y.data

        dataset_train = data_utils.TensorDataset(X, y)
        train_loader = data_utils.DataLoader(dataset_train, batch_size = batch_size, shuffle = True)
        
        y_pred_test = self(X_test)
        loss = get_criterion("mse")(y_pred_test, y_test)
        loss_list.append(loss.data[0])
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
        if self.is_cuda:
            reg = reg.cuda()
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

    
class Generative_Net_Conv(nn.Module):
    def __init__(
        self, 
        input_channels,
        latent_size,
        W_struct_param_list,
        b_struct_param_list,
        struct_param_model,
        num_context_neurons = 0, 
        settings_generative = {"activation": "leakyRelu"}, 
        settings_model = {"activation": "leakyRelu"}, 
        learnable_latent_param = False,
        last_layer_linear = True,
        is_cuda = False,
        ):
        super(Generative_Net_Conv, self).__init__()
        assert len(struct_param_model) == len(W_struct_param_list) == len(b_struct_param_list)
        self.input_channels = input_channels
        self.latent_size = latent_size
        self.W_struct_param_list = W_struct_param_list
        self.b_struct_param_list = b_struct_param_list
        self.struct_param_model = struct_param_model
        self.num_context_neurons = num_context_neurons
        self.settings_generative = settings_generative
        self.settings_model = settings_model
        self.learnable_latent_param = learnable_latent_param
        self.last_layer_linear = last_layer_linear
        self.is_cuda = is_cuda
        self.param_available = ["Conv2d", "ConvTranspose2d", "BatchNorm2d", "Simple_Layer"]

        for i in range(len(self.struct_param_model)):
            if self.struct_param_model[i][1] in self.param_available:
                setattr(self, "W_gen_{0}".format(i), Net(input_size = self.latent_size + num_context_neurons, struct_param = self.W_struct_param_list[i], settings = self.settings_generative, is_cuda = is_cuda))
                setattr(self, "b_gen_{0}".format(i), Net(input_size = self.latent_size + num_context_neurons, struct_param = self.b_struct_param_list[i], settings = self.settings_generative, is_cuda = is_cuda))  
        # Setting up latent param and context param:
        self.latent_param = nn.Parameter(torch.randn(1, self.latent_size)) if learnable_latent_param else None
        if self.num_context_neurons > 0:
            self.context = nn.Parameter(torch.randn(1, self.num_context_neurons))
        if self.is_cuda:
            self.cuda()

    def init_weights_bias(self, latent_param):
        if self.num_context_neurons > 0:
            latent_param = torch.cat([latent_param, self.context], 1)
        for i in range(len(self.struct_param_model)):
            if self.struct_param_model[i][1] in self.param_available:
                setattr(self, "W_{0}".format(i), (getattr(self, "W_gen_{0}".format(i))(latent_param)).squeeze(0))
                setattr(self, "b_{0}".format(i), getattr(self, "b_gen_{0}".format(i))(latent_param)) 
   
    def forward(self, input, latent_param = None):
        if latent_param is None:
            latent_param = self.latent_param
        self.init_weights_bias(latent_param)
        output = input
        for i in range(len(self.struct_param_model)):
            layer_struct_param = self.struct_param_model[i]
            num_neurons_prev = self.struct_param_model[i - 1][0] if i > 0 else self.input_channels
            num_neurons = layer_struct_param[0]
            layer_type = layer_struct_param[1]
            layer_settings = layer_struct_param[2]

            if layer_type in ["Conv2d", "ConvTranspose2d"]:
                kernel_size = layer_settings["kernel_size"]
                weight = getattr(self, "W_{0}".format(i)).view(num_neurons, num_neurons_prev, kernel_size, kernel_size)
                bias = getattr(self, "b_{0}".format(i)).view(-1)
                if layer_type == "Conv2d":
                    output = F.conv2d(output,
                                      weight = weight,
                                      bias = bias,
                                      stride = layer_settings["stride"],
                                      padding = layer_settings["padding"] if "padding" in layer_settings else 0,
                                      dilation = layer_settings["dilation"] if "dilation" in layer_settings else 1,
                                     )
                elif layer_type == "Conv2dTranspose":
                    output = F.conv_transpose2d(output,
                                      weight = weight,
                                      bias = bias,
                                      stride = layer_settings["stride"],
                                      padding = layer_settings["padding"] if "padding" in layer_settings else 0,
                                      dilation = layer_settings["dilation"] if "dilation" in layer_settings else 1,
                                     )
                else:
                    raise Exception("Layer_type {0} not valid!".format(layer_type))
            elif layer_type == "BatchNorm2d":
                weight = getattr(self, "W_{0}".format(i)).view(-1)
                bias = getattr(self, "b_{0}".format(i)).view(-1)
                running_mean = torch.zeros(num_neurons)
                running_var = torch.ones(num_neurons)
                if self.is_cuda:
                    running_mean = running_mean.cuda()
                    running_var = running_var.cuda()
                # Here we are using a hack, letting momentum = 1 to avoid calculating the running statistics:
                output = F.batch_norm(output,
                                      running_mean = running_mean,
                                      running_var = running_var, 
                                      weight = weight, 
                                      bias = bias,
                                     )
            elif layer_type == "Simple_Layer":
                weight = getattr(self, "W_{0}".format(i)).view(num_neurons_prev, num_neurons)
                bias = getattr(self, "b_{0}".format(i)).view(-1)
                output = torch.matmul(output, weight) + bias
            elif layer_type == "MaxPool2d":
                output = F.max_pool2d(output,
                                      layer_settings["kernel_size"],
                                      stride = layer_settings["stride"] if "stride" in layer_settings else None,
                                      padding = layer_settings["padding"] if "padding" in layer_settings else 0,
                                      return_indices = layer_settings["return_indices"] if "return_indices" in layer_settings else False,
                                     )
            elif layer_type == "MaxUnpool2d":
                output = F.max_unpool2d(output,
                                      layer_settings["kernel_size"],
                                      stride = layer_settings["stride"] if "stride" in layer_settings else None,
                                      padding = layer_settings["padding"] if "padding" in layer_settings else 0,
                                     )
            elif layer_type == "Upsample":
                output = F.upsample(output,
                                    scale_factor = layer_settings["scale_factor"],
                                    mode = layer_settings["mode"] if "mode" in layer_settings else "nearest",
                                   )
            else:
                raise Exception("Layer_type {0} not valid!".format(layer_type))

            # Activation:
            if i == len(self.struct_param_model) - 1 and hasattr(self, "last_layer_linear") and self.last_layer_linear:
                activation = "linear"
            else:
                if "activation" in layer_settings:
                    activation = layer_settings["activation"]
                else:
                    activation = self.settings_model["activation"] if "activation" in self.settings_model else "leakyRelu"
            
            output = get_activation(activation)(output)
        return output

    @property
    def model_dict(self):
        model_dict = {"type": "Generative_Net_Conv"}
        model_dict["input_channels"] = self.input_channels
        model_dict["latent_size"] = self.latent_size
        model_dict["W_struct_param_list"] = self.W_struct_param_list
        model_dict["b_struct_param_list"] = self.b_struct_param_list
        model_dict["struct_param_model"] = self.struct_param_model
        model_dict["num_context_neurons"] = self.num_context_neurons
        model_dict["settings_generative"] = self.settings_generative
        model_dict["settings_model"] = self.settings_model
        model_dict["param_available"] = self.param_available
        model_dict["learnable_latent_param"] = self.learnable_latent_param
        model_dict["last_layer_linear"] = self.last_layer_linear
        for i in range(len(self.struct_param_model)):
            if self.struct_param_model[i][1] in self.param_available:
                model_dict["W_gen_{0}".format(i)] = getattr(self, "W_gen_{0}".format(i)).model_dict
                model_dict["b_gen_{0}".format(i)] = getattr(self, "b_gen_{0}".format(i)).model_dict
        if self.latent_param is None:
            model_dict["latent_param"] = None
        else:
            model_dict["latent_param"] = self.latent_param.cpu().data.numpy() if self.is_cuda else self.latent_param.data.numpy()
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
    
    
    def latent_param_quick_learn(self, X, y, validation_data, loss_core = "huber", epochs = 10, batch_size = 128, lr = 1e-2, optim_type = "LBFGS"):
        assert self.learnable_latent_param is True, "To quick-learn latent_param, you must set learnable_latent_param as True!"
        self.latent_param_optimizer = get_optimizer(optim_type = optim_type, lr = lr, parameters = [self.latent_param])
        self.criterion = get_criterion(loss_core)
        loss_list = []
        X_test, y_test = validation_data
        batch_size = min(batch_size, len(X))
        if isinstance(X, Variable):
            X = X.data
        if isinstance(y, Variable):
            y = y.data

        dataset_train = data_utils.TensorDataset(X, y)
        train_loader = data_utils.DataLoader(dataset_train, batch_size = batch_size, shuffle = True)
        
        y_pred_test = self(X_test)
        loss = get_criterion("mse")(y_pred_test, y_test)
        loss_list.append(loss.data[0])
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


    def get_regularization(self, source = ["weight", "bias"], mode = "L1"):
        reg = Variable(torch.FloatTensor(np.array([0])), requires_grad = False)
        if self.is_cuda:
            reg = reg.cuda()
        for reg_type in source:
            for i in range(len(self.struct_param_model)):
                if self.struct_param_model[i][1] not in self.param_available:
                    continue
                if reg_type == "weight":
                    if mode == "L1":
                        reg = reg + getattr(self, "W_{0}".format(i)).abs().sum()
                    else:
                        raise
                elif reg_type == "bias":
                    if mode == "L1":
                        reg = reg + getattr(self, "b_{0}".format(i)).abs().sum()
                    else:
                        raise
                elif reg_type == "W_gen":
                    reg = reg + getattr(self, "W_gen_{0}".format(i)).get_regularization(source = source, mode = mode)
                elif reg_type == "b_gen":
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
  


def forward(model, X):
    """General function for applying the same model at multiple time steps"""
    output_list = []
    for i in range(X.size(1)):
        output = model(X[:,i:i+1,...])
        if isinstance(output, tuple):
            output = output[0]
        output_list.append(output)
    output_seq = torch.cat(output_list, 1)
    return output_seq


def get_forward_pred(predictor, latent, forward_steps, latent_param = None, is_time_series = True, jump_step = 2, is_flatten = False, oracle_size = None):
    """Applying the same model to roll out several time steps"""
    if not is_time_series:
        if latent_param is None:
            pred_list = predictor(latent)
        else:
            pred_list = predictor(latent, latent_param)      
    else:
        max_forward_steps = max(forward_steps)
        current_latent = latent
        pred_list = []
        for i in range(1, max_forward_steps + 1):
            if latent_param is None:
                current_pred = predictor(current_latent)
            else:
                current_pred = predictor(current_latent, latent_param)
            pred_list.append(current_pred)
            if oracle_size is None:
                current_latent = torch.cat([current_latent[:,jump_step:], current_pred], 1)
            else:
                current_latent = torch.cat([current_latent[:,jump_step:-oracle_size], current_pred, current_latent[:,-oracle_size:]], 1)
        pred_list = torch.cat(pred_list, 1)
        pred_list = pred_list.view(pred_list.size(0), -1, 2)
        forward_steps_idx = torch.LongTensor(np.array(forward_steps) - 1)
        if predictor.is_cuda:
            forward_steps_idx = forward_steps_idx.cuda()
        pred_list = pred_list[:, forward_steps_idx]
        if is_flatten:
            pred_list = pred_list.view(pred_list.size(0), -1)
    return pred_list


def get_autoencoder_losses(conv_encoder, predictor, X_motion, y_motion, forward_steps):
    """Getting autoencoder loss"""
    latent = forward(conv_encoder.encode, X_motion).view(X_motion.size(0), -1, 2)
    latent_pred = get_forward_pred(predictor, latent, forward_steps = forward_steps, is_time_series = True)
    
    pred_recons = forward(conv_encoder.decode, latent_pred)
    recons = forward(conv_encoder.decode, latent)
    loss_auxiliary = nn.MSELoss()(recons, X_motion)
    forward_steps_idx = torch.LongTensor(np.array(forward_steps) - 1)
    if predictor.is_cuda:
        forward_steps_idx = forward_steps_idx.cuda()
    y_motion = y_motion[:, forward_steps_idx]
    loss_pred_recons = nn.MSELoss()(pred_recons, y_motion)
    return loss_auxiliary, loss_pred_recons, pred_recons


def get_rollout_pred_loss(conv_encoder, predictor, X_motion, y_motion, max_step, isplot = True):
    """Getting the loss for multiple forward steps"""
    step_list = []
    loss_step_list = []
    for i in range(1, max_step + 1):
        _, loss_pred_recons, _ = get_losses(conv_encoder, predictor, X_motion, y_motion, forward_steps = [i])
        step_list.append(i)
        loss_step_list.append(loss_pred_recons.data[0])
    if isplot:
        plt.plot(step_list, loss_step_list)
        plt.show()
        plt.clf()
        plt.close()
    return step_list, loss_step_list


class Loss_with_autoencoder(nn.Module):
    def __init__(self, core, forward_steps, aux_coeff = 0.5, is_cuda = False):
        super(Loss_with_autoencoder, self).__init__()
        self.core = core
        self.aux_coeff = aux_coeff
        self.loss_fun = get_criterion(self.core)
        self.is_cuda = is_cuda
        self.forward_steps_idx = torch.LongTensor(np.array(forward_steps) - 1)
        if self.is_cuda:
            self.forward_steps_idx = self.forward_steps_idx.cuda()
    
    def forward(self, X_latent, y_latent_pred, X_train_obs, y_train_obs, autoencoder, loss_fun = None, verbose = False, oracle_size = None):
        if oracle_size is not None:
            X_latent = X_latent[:, : -oracle_size].contiguous()
        X_latent = X_latent.view(X_latent.size(0), -1, 2)
        recons = forward(autoencoder.decode, X_latent)
        pred_recons = forward(autoencoder.decode, y_latent_pred.view(y_latent_pred.size(0), -1, 2))
        if loss_fun is None:
            loss_fun = self.loss_fun
        loss_auxilliary = loss_fun(recons, X_train_obs)
        loss_pred = loss_fun(pred_recons, y_train_obs[:, self.forward_steps_idx])
        if verbose:
            print("loss_aux: {0:.6f}\t loss_pred: {1:.6f}".format(loss_auxilliary.data[0], loss_pred.data[0]))
        return loss_pred + loss_auxilliary * self.aux_coeff


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
    if mu.is_cuda:
        eps = eps.cuda()
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
            W_init = (getattr(generative_Net, "W_gen_{0}".format(i))(statistics)).squeeze(0)
            b_init = getattr(generative_Net, "b_gen_{0}".format(i))(statistics)
            if generative_Net.is_cuda:
                W_init = W_init.cpu()
                b_init = b_init.cpu()
            W_init_list.append(W_init.data.numpy())
            b_init_list.append(b_init.data.numpy()[0])
        else:
            W_init_list.append(None)
            b_init_list.append(None)
    if generative_Net.last_layer_linear is True:
        struct_param[-1][2]["activation"] = "linear"
    return Net(input_size = input_size, struct_param = struct_param, W_init_list = W_init_list, b_init_list = b_init_list, settings = generative_Net.settings_model, is_cuda = generative_Net.is_cuda)


def get_nets(
    input_size,
    output_size,
    target_size = None,
    main_hidden_neurons = [20, 20],
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
    is_cuda = False,
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
    if target_size is None:
        target_size = output_size
    statistics_Net = Statistics_Net(input_size = input_size + target_size,
                                    pre_pooling_neurons = pre_pooling_neurons,
                                    struct_param_pre = struct_param_pre,
                                    struct_param_post = struct_param_post,
                                    struct_param_post_logvar = struct_param_post_logvar,
                                    pooling = statistics_pooling,
                                    settings = {"activation": activation_statistics},
                                    is_cuda = is_cuda,
                                   )

    # For Generative_Net:
    struct_param_gen_base = [
        [60, layer_type, {}],
        [60, layer_type, {}],
        [60, layer_type, {}],
    ] if struct_param_gen_base is None else struct_param_gen_base
    
    W_struct_param_list = []
    b_struct_param_list = []
    all_neurons = list(main_hidden_neurons) + [output_size]
    for i, num_neurons in enumerate(all_neurons):
        num_neurons_prev = all_neurons[i - 1] if i > 0 else input_size
        struct_param_weight = struct_param_gen_base + [[(num_neurons_prev, num_neurons), layer_type, {"activation": "linear"}]]
        struct_param_bias = struct_param_gen_base + [[num_neurons, layer_type, {"activation": "linear"}]]
        W_struct_param_list.append(struct_param_weight)
        b_struct_param_list.append(struct_param_bias)
    generative_Net = Generative_Net(input_size = statistics_output_neurons,
                                    num_context_neurons = num_context_neurons,
                                    W_struct_param_list = W_struct_param_list,
                                    b_struct_param_list = b_struct_param_list,
                                    settings_generative = {"activation": activation_generative},
                                    settings_model = {"activation": activation_model},
                                    learnable_latent_param = learnable_latent_param,
                                    last_layer_linear = True,
                                    is_cuda = is_cuda,
                                   )
    if is_uncertainty_net:
        generative_Net_logstd = Generative_Net(input_size = statistics_output_neurons,
                                                num_context_neurons = num_context_neurons,
                                                W_struct_param_list = W_struct_param_list,
                                                b_struct_param_list = b_struct_param_list,
                                                settings_generative = {"activation": activation_generative},
                                                settings_model = {"activation": activation_model},
                                                learnable_latent_param = learnable_latent_param,
                                                last_layer_linear = True,
                                                is_cuda = is_cuda,
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


def get_tasks(task_id_list, num_train, num_test, task_settings = {}, is_cuda = False, verbose = False, **kwargs):
    num_tasks = num_train + num_test
    tasks = {}
    for j in range(num_tasks):
        if verbose:
            print(j)
        task_id = np.random.choice(task_id_list)
        num_examples = task_settings["num_examples"] if "num_examples" in task_settings else 2000
        if task_id[:12] == "latent-linear":
            task = get_latent_model_data(task_settings["z_settings"], settings = task_settings, num_examples = num_examples, is_cuda = is_cuda,)
        elif task_id[:10] == "polynomial":
            order = int(task_id.split("-")[1])
            task = get_polynomial_class(task_settings["z_settings"], order = order, settings = task_settings, num_examples = num_examples, is_cuda = is_cuda,)
        elif task_id[:8] == "Legendre":
            order = int(task_id.split("-")[1])
            task = get_Legendre_class(task_settings["z_settings"], order = order, settings = task_settings, num_examples = num_examples, is_cuda = is_cuda,)
        elif task_id[:2] == "M-":
            task_mode = task_id.split("-")[1]
            task = get_master_function(task_settings["z_settings"], mode = task_mode, settings = task_settings, num_examples = num_examples, is_cuda = is_cuda,)
        elif task_id[:2] == "C-":
            task_mode = task_id.split("-")[1]
            task = get_master_function_comparison(mode = task_mode, settings = task_settings, num_examples = num_examples, is_cuda = is_cuda,)
        elif task_id == "bounce-states":
            task = get_bouncing_states(data_format = "states", settings = task_settings, num_examples = num_examples, is_cuda = is_cuda, **kwargs)
        elif task_id == "bounce-images":
            task = get_bouncing_states(data_format = "images", settings = task_settings, num_examples = num_examples, is_cuda = is_cuda, **kwargs)
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


def evaluate(task, master_model = None, model = None, criterion = None, is_time_series = True, oracle_size = None, is_VAE = False, is_regulated_net = False, autoencoder = None, forward_steps = [1], **kwargs):
    if autoencoder is not None:
        forward_steps_idx = torch.LongTensor(np.array(forward_steps) - 1)    
        ((X_train_obs, y_train_obs), (X_test_obs, y_test_obs)), z_info = task
        if X_train_obs.is_cuda:
            forward_steps_idx = forward_steps_idx.cuda()   
        X_train = forward(autoencoder.encode, X_train_obs)
        y_train = forward(autoencoder.encode, y_train_obs[:, forward_steps_idx])
        X_test = forward(autoencoder.encode, X_test_obs)
        y_test = forward(autoencoder.encode, y_test_obs[:, forward_steps_idx])
        if oracle_size is not None:
            z_train = Variable(torch.FloatTensor(np.repeat(np.expand_dims(z_info["z"],0), len(X_train), 0)), requires_grad = False)
            z_test = Variable(torch.FloatTensor(np.repeat(np.expand_dims(z_info["z"],0), len(X_test), 0)), requires_grad = False)
            if X_train.is_cuda:
                z_train = z_train.cuda()
                z_test = z_test.cuda()
            X_train = torch.cat([X_train, z_train], 1)
            X_test = torch.cat([X_test, z_test], 1)
    else:
        ((X_train, y_train), (X_test, y_test)), _ = task

    loss_fun = nn.MSELoss()
    if master_model is not None:
        assert model is None
        if is_VAE:
            statistics_mu, statistics_logvar = master_model.statistics_Net(torch.cat([X_train, y_train], 1))
            statistics_sampled = sample_Gaussian(statistics_mu, statistics_logvar)
            y_pred_sampled = master_model.generative_Net(X_test, statistics_sampled)
            loss_sampled, KLD = criterion(y_pred_sampled, y_test, statistics_mu, statistics_logvar)

            y_pred = master_model.generative_Net(X_test, statistics_mu)
            loss = criterion.criterion(y_pred, y_test)
            mse = loss_fun(y_pred, y_test)
            return loss.data[0], loss_sampled.data[0], mse.data[0], KLD.data[0]
        else:
            if master_model.generative_Net_logstd is None:
                statistics = master_model.statistics_Net(torch.cat([X_train, y_train], 1))
                if is_regulated_net:
                    statistics = get_regulated_statistics(master_model.generative_Net, statistics)
                if autoencoder is not None:
                    master_model.generative_Net.set_latent_param(statistics)
                    y_pred = get_forward_pred(master_model.generative_Net, X_test, forward_steps, is_time_series = is_time_series)
                    loss = criterion(X_test, y_pred, X_test_obs, y_test_obs, autoencoder)
                    mse = criterion(X_test, y_pred, X_test_obs, y_test_obs, autoencoder, loss_fun = loss_fun, verbose = False)
                else:
                    y_pred = get_forward_pred(master_model.generative_Net, X_test, forward_steps, is_time_series = is_time_series, latent_param = statistics, jump_step = 2, is_flatten = True)
                    loss = criterion(y_pred, y_test)
                    mse = loss_fun(y_pred, y_test)  
            else:
                statistics_mu, statistics_logvar = master_model.statistics_Net(torch.cat([X_train, y_train], 1))
                if is_regulated_net:
                    statistics_mu = get_regulated_statistics(master_model.generative_Net, statistics_mu)
                    statistics_logvar = get_regulated_statistics(master_model.generative_Net_logstd, statistics_logvar)
                y_pred = master_model.generative_Net(X_test, statistics_mu)
                y_pred_logstd = master_model.generative_Net_logstd(X_test, statistics_logvar)
                loss = criterion(y_pred, y_test, log_std = y_pred_logstd)
                mse = loss_fun(y_pred, y_test)
            return loss.data[0], loss.data[0], mse.data[0], 0
    else:
        if autoencoder is not None:
            y_pred = get_forward_pred(model, X_test, forward_steps, is_time_series = is_time_series)
            loss = loss_sampled = loss_test_sampled = criterion(X_test, y_pred, X_test_obs, y_test_obs, autoencoder, oracle_size = oracle_size)
            mse = criterion(X_test, y_pred, X_test_obs, y_test_obs, autoencoder, loss_fun = loss_fun, verbose = True, oracle_size = oracle_size)
        else:
            y_pred = get_forward_pred(model, X_test, forward_steps, is_time_series = is_time_series)
            loss = loss_sampled = criterion(y_pred, y_test)
            mse = loss_fun(y_pred, y_test)
        return loss.data[0], loss_sampled.data[0], mse.data[0], 0     


def get_reg(reg_dict, statistics_Net = None, generative_Net = None, autoencoder = None, net = None, is_cuda = False):
    reg = Variable(torch.FloatTensor([0]), requires_grad = False)
    if is_cuda:
        reg = reg.cuda()
    for net_name, reg_info in reg_dict.items():
        if net_name == "statistics_Net":
            reg_net = statistics_Net
        elif net_name == "generative_Net":
            reg_net = generative_Net
        elif net_name == "autoencoder":
            reg_net = autoencoder
        elif net_name == "net":
            reg_net = net
        if isinstance(reg_net, nn.DataParallel):
            reg_net = reg_net.module
        if reg_net is not None:
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


def plot_task_ensembles(tasks, master_model = None, model = None, is_time_series = True, is_oracle = False, is_VAE = False, is_uncertainty_net = False, is_regulated_net = False, autoencoder = None, title = None, isplot = True, **kwargs):
    import matplotlib.pyplot as plt
    statistics_list = []
    z_list = []
    for task_key, task in tasks.items():
        if autoencoder is not None:
            forward_steps = kwargs["forward_steps"]
            forward_steps_idx = torch.LongTensor(np.array(forward_steps) - 1)
            ((X_train_obs, y_train_obs), (X_test_obs, y_test_obs)), info = task
            if X_test_obs.is_cuda:
                forward_steps_idx = forward_steps_idx.cuda()
            X_train = forward(autoencoder.encode, X_train_obs)
            y_train = forward(autoencoder.encode, y_train_obs[:, forward_steps_idx])
            X_test = forward(autoencoder.encode, X_test_obs)
            y_test = forward(autoencoder.encode, y_test_obs[:, forward_steps_idx])
            if is_oracle:
                z_train = Variable(torch.FloatTensor(np.repeat(np.expand_dims(info["z"],0), len(X_train), 0)), requires_grad = False)
                z_test = Variable(torch.FloatTensor(np.repeat(np.expand_dims(info["z"],0), len(X_test), 0)), requires_grad = False)
                if X_train.is_cuda:
                    z_train = z_train.cuda()
                    z_test = z_test.cuda()
                X_train = torch.cat([X_train, z_train], 1)
                X_test = torch.cat([X_test, z_test], 1)
        else:
            ((X_train, y_train), (X_test, y_test)), info = task
            
        if master_model is not None:
            results = master_model.get_predictions(X_test = X_test, X_train = X_train, y_train = y_train, is_time_series = is_time_series,
                                                  is_VAE = is_VAE, is_uncertainty_net = is_uncertainty_net, is_regulated_net = is_regulated_net)
            statistics_list.append(to_np_array(results["statistics"])[0])
        else:
            results = {}
            results["y_pred"] = model(X_test)
            statistics_list.append([0, 0])
        z_list.append(info["z"])
        if isplot:
            plt.plot(to_np_array(y_test)[:,0], to_np_array(results["y_pred"])[:,0], ".", markersize = 1, alpha = 0.5)
    if isplot:
        if title is not None:
            plt.title(title)
        plt.show()
        plt.clf()
        plt.close()
    return np.array(statistics_list), np.array(z_list)


def plot_individual_tasks(tasks, master_model = None, model = None, max_plots = 24, is_time_series = True,
                          is_VAE = False, is_uncertainty_net = False, is_regulated_net = False, xlim = (-4, 4), sample_times = None, is_oracle = False):
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
    is_cuda = tasks[list(tasks.keys())[0]][0][0][0].is_cuda
    if xlim is not None:
        X_linspace = Variable(torch.linspace(xlim[0], xlim[1], 200).unsqueeze(1))
        if is_cuda:
            X_linspace = X_linspace.cuda()
    for task_id, task in tasks.items():
        ((X_train, y_train), (X_test, y_test)), info = task
        
        if is_oracle:
            input_size = X_test.size(1) - len(info["z"].squeeze())
        else:
            input_size = X_test.size(1)
        chosen_dim = np.random.choice(range(input_size))
        
        if master_model is not None:
            results = master_model.get_predictions(X_test = X_linspace, X_train = X_train, y_train = y_train, is_time_series = is_time_series, 
                                                  is_VAE = is_VAE, is_uncertainty_net = is_uncertainty_net, is_regulated_net = is_regulated_net)
            statistics_list.append(to_np_array(results["statistics"]))
        else:
            results = {}
            if is_oracle:
                z = to_Variable(np.repeat(np.expand_dims(info["z"], 0), len(X_linspace), 0), is_cuda = is_cuda)
                X_linspace_feed = torch.cat([X_linspace, z], 1)
            else:
                X_linspace_feed = X_linspace
            results["y_pred"] = model(X_linspace_feed)
            statistics_list.append([0,0])
        
        if task_id not in chosen_id:
            continue
        
        ax = fig.add_subplot(num_rows, num_columns, i + 1)
        if sample_times is None:
            if input_size == 1:
                X_linspace_numpy, y_pred_numpy = to_np_array(X_linspace, results["y_pred"])
                ax.plot(X_linspace_numpy[:, chosen_dim], y_pred_numpy.squeeze(), "-r", markersize = 3, label = "pred")
                if master_model is not None and master_model.generative_Net_logstd is not None:
                    y_pred_std = torch.exp(results["y_pred_logstd"])
                    y_pred_std_numpy = to_np_array(y_pred_std)
                    ax.fill_between(X_linspace_numpy[:, chosen_dim], (y_pred_numpy - y_pred_std_numpy).squeeze(), (y_pred_numpy + y_pred_std_numpy).squeeze(), color = "r", alpha = 0.3)
        else:
            y_pred_list = []
            for j in range(sample_times):
                statistics_sampled = sample_Gaussian(results["statistics"], results["statistics_logvar"])
                y_pred = master_model.generative_Net(X_linspace, results["statistics"])
                y_pred_list.append(to_np_array(y_pred))
            y_pred_list = np.concatenate(y_pred_list, 1)
            y_pred_mean = np.mean(y_pred_list, 1)
            y_pred_std = np.std(y_pred_list, 1)
            ax.errorbar(to_np_array(X_linspace)[:, chosen_dim], to_np_array(y_pred_mean), yerr = to_np_array(y_pred_std), fmt="-r", markersize = 3, label = "pred")
        ax.plot(to_np_array(X_test)[:, chosen_dim], to_np_array(y_test).squeeze(), ".", markersize = 3, label = "target")
        
        ax.set_xlabel("x_{0}".format(chosen_dim))
        ax.set_ylabel("y")
        ax.set_title(task_id)
        i += 1
    plt.show()
    plt.clf()
    plt.close()
    return [statistics_list]


def plot_individual_tasks_bounce(
    tasks,
    num_examples_show = 30,
    num_tasks_show = 6,
    master_model = None,
    model = None,
    autoencoder = None,
    num_shots = None,
    highlight_top = None,
    valid_input_dims = None,
    target_forward_steps = 1,
    eval_forward_steps = 1,
    **kwargs
    ):
    import matplotlib.pylab as plt
    fig = plt.figure(figsize = (25, num_tasks_show / 3 * 8))
    plt.subplots_adjust(hspace = 0.4)
    tasks_key_show = np.random.choice(list(tasks.keys()), min(num_tasks_show, len(tasks)), replace = False)
    for k, task_key in enumerate(tasks_key_show):
        if autoencoder is not None:
            forward_steps = list(range(1, eval_forward_steps + 1))
            forward_steps_idx = torch.LongTensor(np.array(forward_steps) - 1)
            if autoencoder.is_cuda:
                forward_steps_idx = forward_steps_idx.cuda()
            ((X_train_obs, y_train_obs), (X_test_obs, y_test_obs)), _ = tasks[task_key]
            X_train = forward(autoencoder.encode, X_train_obs)
            y_train = forward(autoencoder.encode, y_train_obs[:, forward_steps_idx])
            X_test = forward(autoencoder.encode, X_test_obs)
            y_test = forward(autoencoder.encode, y_test_obs[:, forward_steps_idx])
        else:
            ((X_train, y_train), (X_test, y_test)), _ = tasks[task_key]
        num_steps = int(X_test.size(1) / 2)
        is_cuda = X_train.is_cuda
        X_test_numpy, y_test_numpy = to_np_array(X_test, y_test)
        if len(X_test_numpy.shape) == 2:
            X_test_numpy = X_test_numpy.reshape(-1, num_steps, 2)
            y_test_numpy = y_test_numpy.reshape(-1, int(y_test_numpy.shape[1] / 2), 2)

        # Get highlighted examples:
        if highlight_top is not None:
            relevance_train = get_relevance(X_train, y_train, master_model.statistics_Net)
            X_sorted, y_sorted, relevance_sorted = sort_datapoints(X_train, y_train, relevance_train, top = highlight_top)
            if len(X_sorted.shape) == 2:
                X_sorted = X_sorted.view(-1, num_steps, 2)
                y_sorted = y_sorted.view(-1, int(y_sorted.shape[1] / 2), 2)
            X_sorted, y_sorted = to_np_array(X_sorted, y_sorted)

        # Get model prediction:
        if master_model is not None:
            if num_shots is None:
                statistics = master_model.statistics_Net.forward_inputs(X_train, y_train[:, :target_forward_steps * 2])
            else:
                idx = torch.LongTensor(np.random.choice(range(len(X_train)), min(len(X_train), num_shots), replace = False))
                if is_cuda:
                    idx = idx.cuda()
                statistics = master_model.statistics_Net.forward_inputs(X_train[idx], y_train[idx, :target_forward_steps * 2])
            if isinstance(statistics, tuple):
                statistics = statistics[0]

            master_model.generative_Net.set_latent_param(statistics)
            model_core = master_model.generative_Net

            # Prediction for highlighted examples:
            if highlight_top is not None:
                y_sorted_pred = model_core(to_Variable(X_sorted.reshape(X_sorted.shape[0], -1), is_cuda = is_cuda))
                y_sorted_pred = to_np_array(y_sorted_pred)
                if len(y_sorted_pred.shape) == 2:
                    y_sorted_pred = y_sorted_pred.reshape(-1, int(y_sorted_pred.shape[1] / 2), 2)
        else:
            assert model is not None
            model_core = model

        preds = predict_forward(model_core, X_test, num_forward_steps = eval_forward_steps)
        y_pred_numpy = to_np_array(reshape_time_series(preds))

        # Plotting:
        ax = fig.add_subplot(int(np.ceil(num_tasks_show / float(3))), 3, k + 1)
        for i in range(len(X_test_numpy)):
            if i >= num_examples_show:
                break
            x_ele = X_test_numpy[i]
            if valid_input_dims is not None:
                x_ele = x_ele[:int(valid_input_dims / 2), :]
            y_ele = y_test_numpy[i]
            ax.plot(np.concatenate((x_ele[:,0], y_ele[:,0])), np.concatenate((x_ele[:,1], y_ele[:,1])), ".-", color = COLOR_LIST[i % len(COLOR_LIST)], zorder = -1)
            ax.scatter(y_ele[:,0], y_ele[:,1], s = np.linspace(10, 20, len(y_ele[:,0])), marker = "o", color = "r", zorder = 2)
            ax.set_title(task_key)
            if master_model is not None or model is not None:
                y_pred_ele = y_pred_numpy[i]
                ax.plot(np.concatenate((x_ele[:,0], y_pred_ele[:,0])), np.concatenate((x_ele[:,1], y_pred_ele[:,1])), ".--", color = COLOR_LIST[i % len(COLOR_LIST)], zorder = -1)
                ax.scatter(y_pred_ele[:,0], y_pred_ele[:,1], s = np.linspace(10, 20, len(y_ele[:,0])), marker = "o", color = "b", zorder = 2)

        # Plotting highlighted examples:
        if highlight_top is not None:
            for i in range(highlight_top):
                x_ele = X_sorted[i]
                y_ele = y_sorted[i]
                ax.plot(np.concatenate((x_ele[:,0], y_ele[:,0])), np.concatenate((x_ele[:,1], y_ele[:,1])), ".-", color = "k", zorder = -1)
                ax.scatter(y_ele[:,0], y_ele[:,1], s = np.linspace(10, 20, len(y_ele[:,0])), marker = "o", color = "r", zorder = 2)
                ax.set_title(task_key)
                if master_model is not None or model is not None:
                    y_pred_ele = y_sorted_pred[i]
                    ax.plot(np.concatenate((x_ele[:,0], y_pred_ele[:,0])), np.concatenate((x_ele[:,1], y_pred_ele[:,1])), ".--", color = "k", zorder = -1)
                    ax.scatter(y_pred_ele[:,0], y_pred_ele[:,1], s = np.linspace(10, 20, len(y_ele[:,0])), marker = "o", color = "k", zorder = 2)

    plt.show()
    plt.clf()
    plt.close()


def plot_few_shot_loss(master_model, tasks, isplot = True, is_time_series = True, autoencoder = None, min_shots = None, forward_steps = [1], **kwargs):
    if master_model is None:
        return []
    num_shots_list = [10, 20, 30, 40, 50, 70, 100, 200, 300, 500, 1000]
    mse_list_whole = []
    for task_key, task in tasks.items():
        mse_list = []
        if autoencoder is not None:
            forward_steps_idx = torch.LongTensor(np.array(forward_steps) - 1)
            if autoencoder.is_cuda:
                forward_steps_idx = forward_steps_idx.cuda()
            ((X_train_obs, y_train_obs), (X_test_obs, y_test_obs)), _ = task
            X_train = forward(autoencoder.encode, X_train_obs)
            y_train = forward(autoencoder.encode, y_train_obs[:, forward_steps_idx])
            X_test = forward(autoencoder.encode, X_test_obs)
            y_test = forward(autoencoder.encode, y_test_obs[:, forward_steps_idx])
        else:
            ((X_train, y_train), (X_test, y_test)), _ = task
        is_cuda = X_train.is_cuda
        for num_shots in num_shots_list:
            if num_shots > len(X_train):
                continue
            if min_shots is not None:
                if num_shots < min_shots:
                    continue
            idx = torch.LongTensor(np.random.choice(range(len(X_train)), num_shots, replace = False))
            if is_cuda:
                idx = idx.cuda()
            X_few_shot = X_train[idx]
            y_few_shot = y_train[idx]
            statistics = master_model.statistics_Net.forward_inputs(X_few_shot, y_few_shot)
            if isinstance(statistics, tuple):
                statistics = statistics[0]
            if autoencoder is not None:
                master_model.generative_Net.set_latent_param(statistics)
                y_pred = get_forward_pred(master_model.generative_Net, X_test, forward_steps, is_time_series = is_time_series)
                mse = kwargs["criterion"](X_test, y_pred, X_test_obs, y_test_obs, autoencoder, loss_fun = nn.MSELoss()).data[0]
            else:
                y_test_pred = get_forward_pred(master_model.generative_Net, X_test, forward_steps, is_time_series = is_time_series, latent_param = statistics, jump_step = 2, is_flatten = True)
                mse = nn.MSELoss()(y_test_pred, y_test).data[0]
            mse_list.append(mse)
        mse_list_whole.append(mse_list)
    mse_list_whole = np.array(mse_list_whole)
    mse_mean = mse_list_whole.mean(0)
    mse_std = mse_list_whole.std(0)
    if isplot:
        import matplotlib.pylab as plt
        plt.figure(figsize = (8,6))
        plt.errorbar(num_shots_list[:len(mse_mean)], mse_mean, mse_std, fmt = "o")
        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_xlabel("number of shots")
        ax.set_ylabel("mse")
        plt.show()

        plt.figure(figsize = (8,6))
        plt.errorbar(num_shots_list[:len(mse_mean)], mse_mean, mse_std, fmt = "o")
        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of shots")
        ax.set_ylabel("mse")
        plt.show()
        plt.clf()
        plt.close()
    return mse_list_whole


def plot_quick_learn_performance(models, tasks, learning_type = "clone_net", is_time_series = True, forward_steps = [1], loss_core = "huber", epochs = 50, lr = 1e-3, batch_size = 128, optim_type = "adam", isplot = True, scale = "normal"):
    if not isinstance(models, dict):
        models = {"model_0": models}
    mse_dict_whole = {model_key: [] for model_key in models.keys()}
    for model_key, model in models.items():
        for task_key, task in tasks.items():
            ((X_train, y_train), (X_test, y_test)), _ = task
            if learning_type == "clone_net":
                if model.__class__.__name__ == "Master_Model":
                    model_core = model.get_clone_net(X_train, y_train)
                else:
                    model_core = model
                mse_list = quick_learn(model_core, X_train, y_train, validation_data = (X_test, y_test), forward_steps = forward_steps, is_time_series = is_time_series, loss_core = loss_core, 
                                       batch_size = batch_size, epochs = epochs, lr = lr, optim_type = optim_type)[0]
            elif learning_type == "latent_param":
                mse_list = model.latent_param_quick_learn(X_train, y_train, validation_data = (X_test, y_test), is_time_series = is_time_series, loss_core = loss_core, 
                                                          epochs = epochs, batch_size = batch_size, lr = lr, optim_type = optim_type, reset_latent_param = True)
            else:
                raise
            
            mse_dict_whole[model_key].append(mse_list)
        mse_dict_whole[model_key] = np.array(mse_dict_whole[model_key])
    epoch_list = list(range(epochs + 1))
    if isplot:
        import matplotlib.pylab as plt
        plt.figure(figsize = (8,6))
        for model_key, model in models.items():
            plt.errorbar(epoch_list, mse_dict_whole[model_key].mean(0), mse_dict_whole[model_key].std(0), fmt = "o", label = model_key)
        ax = plt.gca()
        ax.legend()
        if scale == "log":
            ax.set_yscale("log")
        ax.set_xlabel("number of epochs")
        ax.set_ylabel("mse")
        plt.show()
        plt.clf()
        plt.close()
    return mse_dict_whole


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
    fig = plt.figure(figsize = (25, num_rows * 3.2))

    for i in range(z_size):
        ax = fig.add_subplot(num_rows, num_columns, i + 1)
        for j in range(statistics_list.shape[1]):
            ax.plot(z_list[:,i], statistics_list[:,j], color = COLOR_LIST[j], marker = ".", linestyle = 'None', alpha = 0.6, markersize = 2)
            ax.set_title("statistics vs. z_{0}".format(i))
    plt.show()
    plt.clf()
    plt.close()
    
    # Plot coefficient for linear regression:
    info = {}
    if mode == "corrcoef":
        print("statistics (row) vs. z (column) pearsonr correlation coefficient (abs value):")
        cross_corrcoef = get_corrcoef(z_list, statistics_list)
        plot_matrices([np.abs(cross_corrcoef)], title = title)
        print("statistics correlation matrix:")
        self_corrcoef = np.corrcoef(statistics_list, rowvar = False)
        plot_matrices([np.abs(self_corrcoef)])
        print("pca explained variance ratio:")
        pca = PCA()
        pca.fit(statistics_list)
        print(pca.explained_variance_ratio_)
        
        info["cross_corrcoef"] = cross_corrcoef
        info["self_corrcoef"] = self_corrcoef
        info["explained_variance_ratio"] = pca.explained_variance_ratio_     
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
        info["coeff_list"] = coeff_list
    return info


def plot_data_record(data_record, idx = None, is_VAE = False, tasks_train_keys = None, tasks_test_keys = None):
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
                if "tasks_train" in data_record:
                    loss_train_list = [data_record[key][task_key][idx] for task_key in data_record["tasks_train"][0].keys()]
                    loss_test_list = [data_record[key][task_key][-1] for task_key in data_record["tasks_test"][0].keys()]
                else:
                    loss_train_list = [data_record[key][task_key][idx] for task_key in tasks_train_keys]
                    loss_test_list = [data_record[key][task_key][-1] for task_key in tasks_test_keys]
                ax.hist(loss_train_list, bins = 20, density = True, alpha = 0.3, color="b")
                ax.hist(loss_test_list, bins = 20, density = True, alpha = 0.3, color="r")
                ax.axvline(x= np.mean(loss_train_list), c = "b", alpha = 0.6, label = "train_mean")
                ax.axvline(x= np.median(loss_train_list), c = "b", linestyle = "--", alpha = 0.6, label = "train_median")
                ax.axvline(x= np.mean(loss_test_list), c = "r", alpha = 0.6, label = "test_mean")
                ax.axvline(x= np.median(loss_test_list), c = "r", linestyle = "--", alpha = 0.6, label = "test_median")
                ax.legend()
                ax.set_title("Histogram for {0}:".format(key))
    plt.show()
    plt.clf()
    plt.close()


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
    num_examples = 1000,
    isTorch = True,
    is_cuda = False,
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
        if is_cuda:
            X_train = X_train.cuda()
            y_train = y_train.cuda()
            X_test = X_test.cuda()
            y_test = y_test.cuda()
    return ((X_train, y_train), (X_test, y_test)), {"z": z}



def get_polynomial_class(
    z_settings = ["Gaussian", (0, 1)],
    order = 3,
    settings = {},
    num_examples = 1000,
    isTorch = True,
    is_cuda = False,
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
        if is_cuda:
            X_train = X_train.cuda()
            y_train = y_train.cuda()
            X_test = X_test.cuda()
            y_test = y_test.cuda()
    return ((X_train, y_train), (X_test, y_test)), {"z": z}


def get_Legendre_class(
    z_settings = ["Gaussian", (0, 1)],
    order = 3,
    settings = {},
    num_examples = 1000,
    isTorch = True,
    is_cuda = False,
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
        if is_cuda:
            X_train = X_train.cuda()
            y_train = y_train.cuda()
            X_test = X_test.cuda()
            y_test = y_test.cuda()
    return ((X_train, y_train), (X_test, y_test)), {"z": z}


def get_master_function_comparison(
    z_settings = {},
    mode = "sin",
    settings = {},
    num_examples = 1000,
    isTorch = True,
    is_cuda = False,
    ):
    test_size = settings["test_size"] if "test_size" in settings else 0.5
    z_info = {}
    if mode == "sin":
        amp_range = [0.1, 5.0]
        phase_range = [0, np.pi]
        xlim = (-5,5)

        X = np.random.uniform(xlim[0], xlim[1], [num_examples, 1])
        amp = np.random.uniform(amp_range[0], amp_range[1])
        phase = np.random.uniform(phase_range[0], phase_range[1])
        y = amp * np.sin(X - phase)
        z_info["z"] = np.array([amp, phase])
    elif mode == "tanh":
        freq_range = [0.5, 1.5]
        x0_range = [-1, 1]
        amp_range = [1, 2]
        const_range = [-1, 1]
        xlim = (-5,5)
        
        X = np.random.uniform(xlim[0], xlim[1], [num_examples, 1])
        freq = np.random.uniform(freq_range[0], freq_range[1])
        x0 = np.random.uniform(x0_range[0], x0_range[1])
        amp = np.random.uniform(amp_range[0], amp_range[1])
        const = np.random.uniform(const_range[0], const_range[1])
        y = np.tanh((X - x0) * freq) * amp + const
        z_info["z"] = np.array([const, amp, freq, x0])
    else:
        raise

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    if isTorch:
        X_train = Variable(torch.FloatTensor(X_train), requires_grad = False)
        y_train = Variable(torch.FloatTensor(y_train), requires_grad = False)
        X_test = Variable(torch.FloatTensor(X_test), requires_grad = False)
        y_test = Variable(torch.FloatTensor(y_test), requires_grad = False)
        if is_cuda:
            X_train = X_train.cuda()
            y_train = y_train.cuda()
            X_test = X_test.cuda()
            y_test = y_test.cuda()
    return ((X_train, y_train), (X_test, y_test)), z_info


def get_master_function(
    z_settings = ["Gaussian", (0, 1)],
    mode = "sawtooth",
    settings = {},
    num_examples = 1000,
    isTorch = True,
    is_cuda = False,
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
        if is_cuda:
            X_train = X_train.cuda()
            y_train = y_train.cuda()
            X_test = X_test.cuda()
            y_test = y_test.cuda()
    return ((X_train, y_train), (X_test, y_test)), {"z": z}


def get_bouncing_states(settings, num_examples, data_format = "states", is_cuda = False, **kwargs):
    from mela.variational.util_variational import get_env_data
    from mela.settings.a2c_env_settings import ENV_SETTINGS_CHOICE
    render = kwargs["render"] if "render" in kwargs else False
    test_size = settings["test_size"] if "test_size" in settings else 0.2
    env_name = "envBounceStates"
    screen_size = ENV_SETTINGS_CHOICE[env_name]["screen_height"]
    ball_radius = ENV_SETTINGS_CHOICE[env_name]["ball_radius"]
    
    vertex_bottom_left = tuple(np.random.rand(2) * screen_size / 3 + ball_radius)
    vertex_bottom_right = (screen_size - np.random.rand() * screen_size / 3 - ball_radius, np.random.rand() * screen_size / 3 + ball_radius)
    vertex_top_right = tuple(screen_size - np.random.rand(2) * screen_size / 3 - ball_radius)
    vertex_top_left = (np.random.rand() * screen_size / 3 + ball_radius, screen_size - np.random.rand() * screen_size / 3 - ball_radius)
    boundaries = [vertex_bottom_left, vertex_bottom_right, vertex_top_right, vertex_top_left]
    
    ((X_train, y_train), (X_test, y_test), (reflected_train, reflected_test)), info =         get_env_data(
            env_name,
            data_format = data_format,
            num_examples = num_examples,
            test_size = test_size,
            isplot = False,
            is_cuda = False,
            output_dims = (0,1),
            episode_length = 200,
            boundaries = boundaries,
            verbose = True,
            **kwargs
        )
    if is_cuda:
        X_train = X_train.cuda()
        y_train = y_train.cuda()
        X_test = X_test.cuda()
        y_test = y_test.cuda()
    return ((X_train, y_train), (X_test, y_test)), {"z": np.array(boundaries).reshape(-1)}

