
# coding: utf-8

# In[1]:


from __future__ import print_function
import numpy as np
from copy import deepcopy
import itertools
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from AI_scientist.util import plot_matrices
from AI_scientist.pytorch.modules import get_Layer, load_layer_dict
from AI_scientist.pytorch.util_pytorch import softmax, get_activation, get_criterion, get_optimizer, get_full_struct_param


# In[2]:


def load_model_dict(model_dict, is_cuda = False):
    net_type = model_dict["net_type"]
    if net_type == "Net":
        return Net(input_size = model_dict["input_size"],
                   struct_param = model_dict["struct_param"],
                   W_init_list = model_dict["weights"],
                   b_init_list = model_dict["bias"],
                   settings = model_dict["settings"],
                   is_cuda = is_cuda,
                  )
    else:
        raise Exception("net_type {0} not recognized!".format(net_type))


# In[3]:


class Net(nn.Module):
    def __init__(
        self,
        input_size,
        struct_param,
        W_init_list = None,     # initialization for weights
        b_init_list = None,     # initialization for bias
        settings = {},          # Default settings for each layer, if the settings for the layer is not provided in struct_param
        is_cuda = False,
        ):
        super(Net, self).__init__()
        self.input_size = input_size
        self.num_layers = len(struct_param)
        self.W_init_list = W_init_list
        self.b_init_list = b_init_list
        self.settings = deepcopy(settings)
        self.is_cuda = is_cuda
        
        self.init_layers(deepcopy(struct_param))


    @property
    def struct_param(self):
        return [getattr(self, "layer_{0}".format(i)).struct_param for i in range(self.num_layers)]


    def init_layers(self, struct_param):
        for k, layer_struct_param in enumerate(struct_param):
            num_neurons_prev = struct_param[k - 1][0] if k > 0 else self.input_size
            num_neurons = layer_struct_param[0]
            W_init = self.W_init_list[k] if self.W_init_list is not None else None
            b_init = self.b_init_list[k] if self.b_init_list is not None else None

            # Get settings for the current layer:
            layer_settings = deepcopy(self.settings) if bool(self.settings) else {}
            layer_settings.update(layer_struct_param[2])
            layer_settings.pop("snap_dict", None)
            if "snap_dict" in self.settings and k in self.settings["snap_dict"]:
                layer_settings["snap_dict"] = self.settings["snap_dict"][k]

            # Construct layer:
            layer = get_Layer(layer_type = layer_struct_param[1],
                              input_size = num_neurons_prev,
                              output_size = num_neurons,
                              W_init = W_init,
                              b_init = b_init,
                              settings = layer_settings,
                              is_cuda = self.is_cuda,
                             )
            setattr(self, "layer_{0}".format(k), layer)


    def init_bias_with_input(self, input, mode = "std_sqrt", neglect_last_layer = True):
        output = input
        for k, layer_struct_param in enumerate(self.struct_param):
            if neglect_last_layer and k == len(self.struct_param) - 1:
                break
            layer = getattr(self, "layer_{0}".format(k))
            layer.init_bias_with_input(output, mode = mode)
            output = layer(output)


    def forward(self, input, p_dict = None):
        output = input
        for k in range(len(self.struct_param)):
            if p_dict is None:
                output = getattr(self, "layer_{0}".format(k))(output)
            else:
                output = getattr(self, "layer_{0}".format(k))(output, p_dict = p_dict[k])
        return output


    def init_with_p_dict(self, p_dict):
        if p_dict is not None:
            for k in range(self.num_layers):
                if k in p_dict:
                    layer = getattr(self, "layer_{0}".format(k))
                    layer.init_with_p_dict(p_dict[k])


    def get_param_names(self, source):
        param_names_list = []
        for k in range(len(self.struct_param)):
            param_names = getattr(self, "layer_{0}".format(k)).get_param_names(source = source)
            param_names = ["layer_{0}.{1}".format(k, name) for name in param_names]
            param_names_list = param_names_list + param_names
        return param_names_list


    def get_regularization(self, source = ["weight", "bias"], mode = "L1"):
        reg = Variable(torch.FloatTensor([0]), requires_grad = False)
        if self.is_cuda:
            reg = reg.cuda()
        for k in range(len(self.struct_param)):
            layer = getattr(self, "layer_{0}".format(k))
            reg = reg + layer.get_regularization(mode = mode, source = source)
        return reg
    
    
    def simplify(self, X, y, mode = "full", **kwargs):
        new_net, loss_dict = simplify(self, X, y, mode = mode, **kwargs)
        self.__dict__.update(new_net.__dict__)  # Replace the current whole instance with the simplified one
        return loss_dict


    def reset_layer(self, layer_id, layer):
        setattr(self, "layer_{0}".format(layer_id), layer)


    def insert_layer(self, layer_id, layer):
        if layer_id < 0:
            layer_id += self.num_layers
        if layer_id < self.num_layers - 1:
            next_layer = getattr(self, "layer_{0}".format(layer_id + 1))
            if next_layer.struct_param[1] == "Simple_Layer":
                assert next_layer.input_size == layer.output_size, "The inserted layer's output_size {0} must be compatible with next layer_{1}'s input_size {2}!"                    .format(layer.output_size, layer_id + 1, next_layer.input_size)
        for i in range(self.num_layers - 1, layer_id - 1, -1):
            setattr(self, "layer_{0}".format(i + 1), getattr(self, "layer_{0}".format(i)))
        setattr(self, "layer_{0}".format(layer_id), layer)
        self.num_layers += 1
    
    
    def remove_layer(self, layer_id):
        if layer_id < 0:
            layer_id += self.num_layers
        if layer_id < self.num_layers - 1:
            num_neurons_prev = self.struct_param[layer_id - 1][0] if layer_id > 0 else self.input_size
            replaced_layer = getattr(self, "layer_{0}".format(layer_id + 1))
            if replaced_layer.struct_param[1] == "Simple_Layer":
                assert replaced_layer.input_size == num_neurons_prev,                     "After deleting layer_{0}, the replaced layer's input_size {1} must be compatible with previous layer's output neurons {2}!"                        .format(layer_id, replaced_layer.input_size, num_neurons_prev)
        for i in range(layer_id, self.num_layers - 1):
            setattr(self, "layer_{0}".format(i), getattr(self, "layer_{0}".format(i + 1)))
        self.num_layers -= 1


    def prune_neurons(self, layer_id, neuron_ids):
        if layer_id < 0:
            layer_id = self.num_layers + layer_id
        layer = getattr(self, "layer_{0}".format(layer_id))
        layer.prune_output_neurons(neuron_ids)
        self.reset_layer(layer_id, layer)
        if layer_id < self.num_layers - 1:
            next_layer = getattr(self, "layer_{0}".format(layer_id + 1))
            next_layer.prune_input_neurons(neuron_ids)
            self.reset_layer(layer_id + 1, next_layer)


    def add_neurons(self, layer_id, num_neurons, mode = ("imitation", "zeros")):
        if not isinstance(mode, list) and not isinstance(mode, tuple):
            mode = (mode, mode)
        if layer_id < 0:
            layer_id = self.num_layers + layer_id
        layer = getattr(self, "layer_{0}".format(layer_id))
        layer.add_output_neurons(num_neurons, mode = mode[0])
        self.reset_layer(layer_id, layer)
        if layer_id < self.num_layers - 1:
            next_layer = getattr(self, "layer_{0}".format(layer_id + 1))
            next_layer.add_input_neurons(num_neurons, mode = mode[1])
            self.reset_layer(layer_id + 1, next_layer)
    
    
    def split_to_net_ensemble(self, mode = "standardize"):
        num_models = self.struct_param[-1][0]
        model_core = deepcopy(self)
        if mode == "standardize":
            last_layer = getattr(model_core, "layer_{0}".format(model_core.num_layers - 1))
            last_layer.standardize(mode = "b_mean_zero")
        else:
            raise Exception("mode {0} not recognized!".format(mode))
        model_list = [deepcopy(model_core) for i in range(num_models)]
        for i, model in enumerate(model_list):
            to_prune = list(range(num_models))
            to_prune.pop(i)
            model.prune_neurons(-1, to_prune)
        return construct_net_ensemble_from_nets(model_list)
            

    def inspect_operation(self, input_, operation_between):
        output = input_
        for k in range(*operation_between):
            output = getattr(self, "layer_{0}".format(k))(output)
        return output


    def get_weights_bias(self, W_source = None, b_source = None, layer_ids = None, isplot = False, raise_error = True):
        layer_ids = range(len(self.struct_param)) if layer_ids is None else layer_ids
        W_list = []
        b_list = []
        if W_source is not None:
            for k in range(len(self.struct_param)):
                if k in layer_ids:
                    if W_source == "core":
                        try:
                            W, _ = getattr(self, "layer_{0}".format(k)).get_weights_bias()
                        except Exception as e:
                            if raise_error:
                                raise
                            else:
                                print(e)
                            W = np.array([np.NaN])
                    else:
                        raise Exception("W_source '{0}' not recognized!".format(W_source))
                    W_list.append(W)
        
        if b_source is not None:
            for k in range(len(self.struct_param)):
                if k in layer_ids:
                    if b_source == "core":
                        try:
                            _, b = getattr(self, "layer_{0}".format(k)).get_weights_bias()
                        except Exception as e:
                            if raise_error:
                                raise
                            else:
                                print(e)
                            b = np.array([np.NaN])
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

    
    def get_significance(self, W_available_all = None, b_available_all = None, layer_ids = None, isplot = False):
        if W_available_all is not None:
            self.W_available_all = W_available_all
        else:
            self.W_available_all = []
            for k in range(len(self.struct_param)):
                layer = getattr(self, "layer_{0}".format(k))
                if isinstance(layer, SuperNet_Layer):
                    self.W_available_all += layer.W_available
            self.W_available_all = sorted(list(set(self.W_available_all)))
        
        if b_available_all is not None:
            self.b_available_all = b_available_all
        else:
            self.b_available_all = []
            for k in range(len(self.struct_param)):
                layer = getattr(self, "layer_{0}".format(k))
                if isinstance(layer, SuperNet_Layer):
                    self.b_available_all += layer.b_available
            self.b_available_all = sorted(list(set(self.b_available_all)))

        layer_ids = range(len(self.struct_param)) if layer_ids is None else layer_ids
        W_sig_array = -np.ones((len(layer_ids), len(self.W_available_all)))
        b_sig_array = -np.ones((len(layer_ids), len(self.b_available_all)))
        i = 0
        layer_ids_vis = []
        for k in range(len(self.struct_param)):
            if k in layer_ids:
                layer = getattr(self, "layer_{0}".format(k))
                if not isinstance(layer, SuperNet_Layer):
                    continue
                layer_ids_vis.append(str(k))
                if self.is_cuda:
                    W_sig = softmax(layer.W_sig.cpu().data.numpy())
                    b_sig = softmax(layer.b_sig.cpu().data.numpy())
                else:
                    W_sig = softmax(layer.W_sig.data.numpy())
                    b_sig = softmax(layer.b_sig.data.numpy())
                for j, W_type in enumerate(self.W_available_all):
                    try:
                        W_idx = layer.W_available.index(W_type)
                    except:
                        W_idx = None
                    W_sig_array[i, j] = W_sig[W_idx] if W_idx is not None else np.NaN
                for j, b_type in enumerate(self.b_available_all):
                    try:
                        b_idx = layer.b_available.index(b_type)
                    except:
                        b_idx = None
                    b_sig_array[i, j] = b_sig[b_idx] if b_idx is not None else np.NaN
                i += 1
        W_sig_array = W_sig_array[:len(layer_ids_vis)]
        b_sig_array = b_sig_array[:len(layer_ids_vis)]
        layer_ids_print = "layers " + ",".join(layer_ids_vis)
        if isplot:
            plot_matrices([W_sig_array, b_sig_array], x_axis_list = [layer_ids_print] * 2)
        return (W_sig_array, b_sig_array), (self.W_available_all, self.b_available_all, layer_ids_vis)


    def get_selectors(self, layer_ids = None, isplot = False):
        selectors_list = []
        axis_label_list = []
        layer_ids = range(len(self.struct_param)) if layer_ids is None else layer_ids
        for k in range(len(self.struct_param)):
            if k in layer_ids:
                layer = getattr(self, "layer_{0}".format(k))
                if not isinstance(layer, Sneuron_Layer):
                    continue
                selectors = layer.get_selectors()
                selectors_list += selectors
                axis_label_list += ["selector_layer{0}_l".format(k), "selector_layer{0}_r".format(k)]
        if isplot:
            plot_matrices(selectors_list, x_axis_list = axis_label_list)
        return selectors_list, axis_label_list


    def get_snap_dict(self):
        snap_dict = {}
        for k in range(len(self.struct_param)):
            layer = getattr(self, "layer_{0}".format(k))
            if hasattr(layer, "snap_dict"):
                recorded_layer_snap_dict = {}
                for key, item in layer.snap_dict.items():
                    recorded_layer_snap_dict[key] = {"new_value": item["new_value"]}
                if len(recorded_layer_snap_dict) > 0:
                    snap_dict[k] = recorded_layer_snap_dict
        return snap_dict


    def synchronize_settings(self):
        snap_dict = self.get_snap_dict()
        if len(snap_dict) > 0:
            self.settings["snap_dict"] = snap_dict
        return self.settings
    
    
    @property
    def model_dict(self):
        model_dict = {"type": "Net"}
        model_dict["input_size"] = self.input_size
        model_dict["struct_param"] = get_full_struct_param(self.struct_param, self.settings)
        model_dict["weights"], model_dict["bias"] = self.get_weights_bias(W_source = "core", b_source = "core")
        model_dict["settings"] = self.synchronize_settings()
        model_dict["net_type"] = "Net"
        return model_dict


    def load_model_dict(self, model_dict):
        new_net = load_model_dict(model_dict, is_cuda = self.is_cuda)
        self.__dict__.update(new_net.__dict__)


# In[5]:


class RNN_Cell(Net):
    def __init__(
        self,
        input_size,
        hidden_size, 
        struct_param,
        W_init_list = None,     # initialization for weights
        b_init_list = None,     # initialization for bias
        settings = {},          # Default settings for each layer, if the settings for the layer is not provided in struct_param
        is_cuda = False,
        ):
        self.input_size_true = input_size
        self.hidden_size = hidden_size
        assert hidden_size == struct_param[-1][0], "hidden_size must be equal to the num_neurons in the last layer of struct_param!"
        super(RNN_Cell, self).__init__(
            input_size = input_size + hidden_size,
            struct_param = struct_param,
            W_init_list = W_init_list,
            b_init_list = b_init_list,
            settings = settings,
            is_cuda = is_cuda,
        )


    def check_forward_input(self, input):
        if input.size(1) != self.input_size_true:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size_true))


    def check_forward_hidden(self, input, hx, hidden_label=''):
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, input.size(1), self.input_size_true))


    def forward(self, input, hx, **kwargs):
        self.check_forward_input(input)
        self.check_forward_hidden(input, hx)
        output = torch.cat((input, hx), -1)
        for k in range(len(self.struct_param)):
            output = getattr(self, "layer_{0}".format(k))(output)
        return output



class RNN_with_encoder(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 50, struct_param_rnn = None, struct_param_encoder = None, struct_param_decoder = None):
        super(RNN_with_encoder, self).__init__()
        self.input_size = input_size
        self.struct_param_rnn = struct_param_rnn
        self.struct_param_encoder = struct_param_encoder
        self.struct_param_decoder = struct_param_decoder 
        if self.struct_param_encoder is not None:
            self.encoder = Net(input_size = input_size, struct_param = struct_param_encoder)
            input_size_encode = struct_param_encoder[-1][0]
        else:
            input_size_encode = input_size
        if self.struct_param_rnn is None:
            self.rnn1 = nn.RNNCell(input_size_encode, hidden_size)
        else:
            self.rnn1 = RNN_Cell(input_size_encode, hidden_size, struct_param_rnn)
        if self.struct_param_decoder is not None:
            self.decoder = Net(input_size = hidden_size, struct_param = struct_param_decoder)
        self.hidden_size = hidden_size


    def forward(self, input, future = 0):
        outputs = []
        h_t = Variable(torch.zeros(input.size(0), self.hidden_size).float(), requires_grad = False)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            input_t_encode = self.encoder(input_t) if self.struct_param_encoder is not None else input_t
            h_t = self.rnn1(input_t_encode, h_t)
            output = self.decoder(h_t)
            outputs += [output]
        for i in range(future):# if we should predict the future
            # The first self.input_size dimensions are the prediction of next time step (The following dimensions may be uncertainty.):
            input_t_encode = self.encoder(output[...,:self.input_size]) if self.struct_param_encoder is not None else output
            h_t = self.rnn1(input_t_encode, h_t)
            output = self.decoder(h_t)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


    def get_hidden(self, input, future = 0):
        hiddens = []
        h_t = Variable(torch.zeros(input.size(0), self.hidden_size).float(), requires_grad = False)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            input_t_encode = self.encoder(input_t) if self.struct_param_encoder is not None else input_t
            h_t = self.rnn1(input_t_encode, h_t)
            output = self.decoder(h_t)
            hiddens.append(h_t)
        for i in range(future):# if we should predict the future
            input_t_encode = self.encoder(output[...,:self.input_size]) if self.struct_param_encoder is not None else output
            h_t = self.rnn1(input_t_encode, h_t)
            output = self.decoder(h_t)
            hiddens.append(h_t)
        hiddens = torch.stack(hiddens, 1)
        return hiddens


# In[6]:


class ConvNet(nn.Module):
    def __init__(self, input_channels, struct_param, settings = {}, is_cuda = False):
        super(ConvNet, self).__init__()
        self.struct_param = struct_param
        self.settings = settings
        self.num_layers = len(struct_param)
        self.is_cuda = is_cuda
        for i in range(len(self.struct_param)):
            if i > 0:
                if "Pool" not in self.struct_param[i - 1][1] and "Unpool" not in self.struct_param[i - 1][1] and "Upsample" not in self.struct_param[i - 1][1]:
                    num_channels_prev = self.struct_param[i - 1][0]
                else: 
                    num_channels_prev = self.struct_param[i - 2][0]
            else:
                num_channels_prev = input_channels
            num_channels = self.struct_param[i][0]
            layer_type = self.struct_param[i][1]
            layer_settings = self.struct_param[i][2]
            if layer_type == "Conv2d":
                layer = nn.Conv2d(num_channels_prev, 
                                  num_channels,
                                  kernel_size = layer_settings["kernel_size"],
                                  stride = layer_settings["stride"],
                                  padding = layer_settings["padding"] if "padding" in layer_settings else 0,
                                 )
            elif layer_type == "ConvTranspose2d":
                layer = nn.ConvTranspose2d(num_channels_prev,
                                           num_channels,
                                           kernel_size = layer_settings["kernel_size"],
                                           stride = layer_settings["stride"],
                                           padding = layer_settings["padding"] if "padding" in layer_settings else 0,
                                          )
            elif layer_type == "MaxPool2d":
                layer = nn.MaxPool2d(kernel_size = layer_settings["kernel_size"],
                                     stride = layer_settings["stride"] if "stride" in layer_settings else None,
                                     padding = layer_settings["padding"] if "padding" in layer_settings else 0,
                                     return_indices = layer_settings["return_indices"] if "return_indices" in layer_settings else False,
                                    )
            elif layer_type == "MaxUnpool2d":
                layer = nn.MaxUnpool2d(kernel_size = layer_settings["kernel_size"],
                                       stride = layer_settings["stride"] if "stride" in layer_settings else None,
                                       padding = layer_settings["padding"] if "padding" in layer_settings else 0,
                                      )
            elif layer_type == "Upsample":
                layer = nn.Upsample(scale_factor = layer_settings["scale_factor"],
                                    mode = layer_settings["mode"] if "mode" in layer_settings else "nearest",
                                   )
            else:
                raise Exception("layer_type {0} not recognized!".format(layer_type))
            setattr(self, "layer_{0}".format(i), layer)
        if self.is_cuda:
            self.cuda()


    def forward(self, input, indices_list = None):
        output = input
        if indices_list is None:
            indices_list = []
        for i in range(len(self.struct_param)):
            if "Unpool" in self.struct_param[i][1]:
                output_tentative = getattr(self, "layer_{0}".format(i))(output, indices_list.pop(-1))
            else:
                output_tentative = getattr(self, "layer_{0}".format(i))(output)
            if isinstance(output_tentative, tuple):
                output, indices = output_tentative
                indices_list.append(indices)
            else:
                output = output_tentative
            if "activation" in self.struct_param[i][2]:
                activation = self.struct_param[i][2]["activation"]
            else:
                if "activation" in self.settings:
                    activation = self.settings["activation"]
                else:
                    activation = "relu"
                if "Pool" in self.struct_param[i - 1][1] or "Unpool" in self.struct_param[i - 1][1] or "Upsample" in self.struct_param[i - 1][1]:
                    activation = "linear"
            output = get_activation(activation)(output)
        return output, indices_list


    def get_regularization(self, source = ["weight", "bias"], mode = "L1"):
        reg = Variable(torch.FloatTensor([0]), requires_grad = False)
        if self.is_cuda:
            reg = reg.cuda()
        for k in range(self.num_layers):
            layer = getattr(self, "layer_{0}".format(k))
            for source_ele in source:
                if source_ele == "weight":
                    item = layer.weight
                elif source_ele == "bias":
                    item = layer.bias
                if mode == "L1":
                    reg = reg + item.abs().sum()
                elif mode == "L2":
                    reg = reg + (item ** 2).sum()
                else:
                    raise Exception("mode {0} not recognized!".format(mode))
        return reg

