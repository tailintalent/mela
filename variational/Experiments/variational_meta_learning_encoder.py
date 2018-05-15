
# coding: utf-8

# In[1]:


import numpy as np
import pickle
from itertools import chain
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
import itertools
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data_utils

import sys, os
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..', '..'))
    from AI_scientist.settings.filepath import variational_model_PATH, dataset_PATH
    isplot = True
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from AI_scientist.settings.filepath import variational_model_PATH, dataset_PATH
    if dataset_PATH[:2] == "..":
        dataset_PATH = dataset_PATH[3:]
    isplot = False

from AI_scientist.util import plot_matrices, make_dir, get_struct_str, get_args, Early_Stopping, record_data, manifold_embedding
from AI_scientist.pytorch.modules import Simple_Layer
from AI_scientist.pytorch.net import Net, ConvNet
from AI_scientist.pytorch.util_pytorch import Loss_with_uncertainty, get_criterion, to_np_array
from AI_scientist.variational.util_variational import get_torch_tasks
from AI_scientist.variational.variational_meta_learning import Master_Model, Statistics_Net, Generative_Net, load_model_dict, get_regulated_statistics
from AI_scientist.variational.variational_meta_learning import VAE_Loss, sample_Gaussian, clone_net, get_nets, get_tasks, evaluate, get_reg, load_trained_models
from AI_scientist.variational.variational_meta_learning import forward, get_forward_pred, get_rollout_pred_loss, get_autoencoder_losses, Loss_with_autoencoder
from AI_scientist.variational.variational_meta_learning import plot_task_ensembles, plot_individual_tasks, plot_statistics_vs_z, plot_data_record, get_corrcoef
from AI_scientist.variational.variational_meta_learning import plot_few_shot_loss, plot_individual_tasks_bounce
from AI_scientist.variational.variational_meta_learning import get_latent_model_data, get_polynomial_class, get_Legendre_class, get_master_function

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
is_cuda = torch.cuda.is_available()
print("is_cuda: {0}".format(is_cuda))


# In[2]:


def combine_dataset(tasks, num = None):
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    num_examples_select = 100
    for i, task in enumerate(tasks.values()):
        if num is not None and i > num:
            break
        ((X_train, y_train), (X_test, y_test)), info = task
        X_train_list.append(X_train[:num_examples_select])
        y_train_list.append(y_train[:num_examples_select])
        X_test_list.append(X_test[:num_examples_select])
        y_test_list.append(y_test[:num_examples_select])
    X_train_all = torch.cat(X_train_list)
    y_train_all = torch.cat(y_train_list)
    X_test_all = torch.cat(X_test_list)
    y_test_all = torch.cat(y_test_list)
    return (X_train_all, y_train_all), (X_test_all, y_test_all)

def plot_encoding(X, autoencoder, target = "encoding"):
    axis_list = []
    X = X.data
    X = X[:200]
    for i in range(len(X)):
        data = X[i]
        row, column = data.squeeze().nonzero().float().mean(0)
        axis_list.append([row, column])
    axis_list = np.array(axis_list)
    encoding = autoencoder.encode(Variable(X)).cpu().data.numpy().astype(float)
    if target == "encoding":
        print("row:")
        plt.scatter(encoding[:,0], encoding[:,1], s = 0.5 + 3 * axis_list[:,0])
        plt.show()
        print("column:")
        plt.scatter(encoding[:,0], encoding[:,1], s = 0.5 + 3 * axis_list[:,1])
        plt.show()
    elif target == "axis":
        print("row:")
        plt.scatter(axis_list[:,0], axis_list[:,1], s = 0.5 + 5 * encoding[:,0])
        plt.show()
        print("column:")
        plt.scatter(axis_list[:,0], axis_list[:,1], s = 0.5 + 5 * encoding[:,1])
        plt.show()


def plot_tasks(tasks, master_model = None, model = None, autoencoder = None, forward_steps = None, num_tasks = 3):
    task_keys = np.random.choice(list(tasks.keys()), num_tasks, replace = False)
    for i, task_key in enumerate(task_keys):
        task = tasks[task_key]
        ((X_train_obs, y_train_obs), (X_test_obs, y_test_obs)), _ = task
        X_train = forward(autoencoder.encode, X_train_obs)
        y_train = forward(autoencoder.encode, y_train_obs[:, forward_steps_idx])
        X_test = forward(autoencoder.encode, X_test_obs)
        y_test = forward(autoencoder.encode, y_test_obs[:, forward_steps_idx])
        
        # Plotting:
        print("Task {0}:".format(task_key))
        plot_matrices(np.concatenate((to_np_array(X_test_obs[0]), to_np_array(y_test_obs)[:, np.array(forward_steps) - 1][0])))
        if master_model is not None:
            statistics = master_model.statistics_Net.forward_inputs(X_train, y_train)
            latent_pred = get_forward_pred(master_model.generative_Net, X_test, forward_steps, is_time_series = True, latent_param = statistics, jump_step = 2, is_flatten = False)
        else:
            latent_pred = get_forward_pred(model, X_test, forward_steps, is_time_series = True, jump_step = 2, is_flatten = False)
        pred_recons = forward(autoencoder.decode, latent_pred)
        plot_matrices(np.concatenate((to_np_array(forward(autoencoder.decode, X_test.view(X_test.size(0), -1, 2))[0]), to_np_array(pred_recons[0]))))  


class Conv_Autoencoder(nn.Module):
    def __init__(
        self,
        encoder_struct_param,
        decoder_struct_param,
        enc_flatten_size = 512,
        latent_size = (1,2),
        settings = {},
        is_cuda = False,
        ):
        super(Conv_Autoencoder, self).__init__()
        self.enc_flatten_size = enc_flatten_size
        self.latent_size = latent_size
        self.encoder = ConvNet(input_channels = 1, struct_param = encoder_struct_param, settings = settings, is_cuda = is_cuda)
        self.decoder = ConvNet(input_channels = encoder_struct_param[-1][0], struct_param = decoder_struct_param, settings = settings, is_cuda = is_cuda)
        self.enc_fully = Simple_Layer(enc_flatten_size, latent_size, settings = {"activation": "linear"}, is_cuda = is_cuda)
        self.dec_fully = Simple_Layer(latent_size, enc_flatten_size, settings = {"activation": "linear"}, is_cuda = is_cuda)
        self.is_cuda = is_cuda
    
    def encode(self, input):
        enc_hidden, _ = self.encoder(input)
        self.enc_hidden_size = enc_hidden.size()[1:]
        enc_flatten = enc_hidden.view(enc_hidden.size(0), -1)
        latent = self.enc_fully(enc_flatten)
        return latent
    
    def decode(self, latent):
        dec_hidden = self.dec_fully(latent).view(-1, *self.enc_hidden_size)
        output, _ = self.decoder(dec_hidden)
        return output
    
    def forward(self, input):
        latent = self.encode(input)
        return self.decode(latent)
    
    def get_regularization(self, source = ["weight", "bias"], mode = "L1"):
        return self.encoder.get_regularization(source = source, mode = mode) +                 self.decoder.get_regularization(source = source, mode = mode) +                 self.enc_fully.get_regularization(source = source, mode = mode) +                 self.dec_fully.get_regularization(source = source, mode = mode)


def train_epoch_pretrain(train_loader, X_test_all, autoencoder, optimizer_pre, isplot = True):
    for batch_id, X_batch in enumerate(train_loader):
        X_batch = Variable(X_batch)
        optimizer_pre.zero_grad()
        reconstruct = forward(autoencoder, X_batch)
        reg = autoencoder.get_regularization(source = ["weight", "bias"]) * reg_amp_autoencoder
        reg_latent = forward(autoencoder, X_batch).mean() * reg_amp_latent
        loss_train = nn.MSELoss()(reconstruct, X_batch) + reg + reg_latent
        loss_train.backward()
        optimizer_pre.step()

    reconstruct_test = forward(autoencoder, X_test_all)
    loss_test = nn.MSELoss()(reconstruct_test, X_test_all)
    to_stop = early_stopping_pre.monitor(loss_test.data[0])
    print("epoch {0} \tloss_train: {1:.6f}\tloss_test: {2:.6f}\treg: {3:.6f}\treg_latent: {4:.6f}".format(epoch, loss_train.data[0], loss_test.data[0], reg.data[0], reg_latent.data[0]))
    if epoch % 10 == 0 and isplot:
        plot_matrices(X_batch[0].cpu().data.numpy(), images_per_row = 5)
        latent = forward(autoencoder.encode, X_batch)
        print("latent: {0}".format(latent.cpu().data.numpy()[0]))
        plot_matrices(reconstruct[0].cpu().data.numpy(), images_per_row = 5)
        plot_encoding(X_train_all[:,:1], autoencoder, "axis")
    return to_stop


def train_epoch_joint(motion_train_loader, X_motion_test, y_motion_test, conv_encoder, predictor, forward_steps, aux_coeff, isplot = True):
    for batch_id, (X_motion_batch, y_motion_batch) in enumerate(motion_train_loader):
        X_motion_batch = Variable(X_motion_batch)
        y_motion_batch = Variable(y_motion_batch)
        optimizer_motion.zero_grad()
        loss_auxiliary, loss_pred_recons, pred_recons_batch = get_losses(conv_encoder, predictor, X_motion_batch, y_motion_batch, forward_steps = forward_steps)
        reg_conv = conv_encoder.get_regularization(source = ["weight", "bias"]) * reg_amp_conv * reg_multiplier[epoch]
        reg_predictor = predictor.get_regularization(source = ["weight", "bias"]) * reg_amp_predictor * reg_multiplier[epoch]
        loss_train = loss_auxiliary * aux_coeff + loss_pred_recons + reg_conv + reg_predictor
        loss_train.backward()
        optimizer_motion.step()

    loss_auxiliary_test, loss_pred_recons_test, pred_recons_test = get_losses(conv_encoder, predictor, X_motion_test, y_motion_test, forward_steps = forward_steps)
    loss_test = loss_auxiliary_test * aux_coeff + loss_pred_recons_test + reg_conv + reg_predictor
    to_stop = early_stopping_motion.monitor(loss_test.data[0])
    print("epoch {0}\tloss_train: {1:.6f}\tloss_test: {2:.6f}\tloss_aux: {3:.6f}\tloss_pred: {4:.6f}\treg_conv: {5:.6f}\treg_predictor: {6:.6f}".format(
        epoch, loss_train.data[0], loss_test.data[0], loss_auxiliary_test.data[0] * aux_coeff, loss_pred_recons_test.data[0], reg_conv.data[0], reg_predictor.data[0]))
    if epoch % 10 == 0 and isplot:
        print("epoch {0}:".format(epoch))
        plot_matrices(np.concatenate((X_motion_batch[0].cpu().data.numpy(), y_motion_batch[:, torch.LongTensor(np.array(forward_steps) - 1).cuda()][0].cpu().data.numpy())))
        plot_matrices(np.concatenate((forward(conv_encoder, X_motion_batch)[0].cpu().data.numpy(), pred_recons_batch[0].cpu().data.numpy())))
        print("encoding:")
        plot_encoding(X_motion_batch[:,:1].contiguous(), conv_encoder, target = "encoding")
        print("axis:")
        plot_encoding(X_motion_batch[:,:1].contiguous(), conv_encoder, target = "axis")
        print("\n\n")
    return to_stop


# ## Pretrain conv-autoencoder:

# In[3]:


# encoder_struct_param = [
#     [32, "Conv2d", {"kernel_size": 3, "stride": 2}],
#     [32, "Conv2d", {"kernel_size": 3, "stride": 2}],
#     [32, "Conv2d", {"kernel_size": 3, "stride": 2}],
# ]
# settings = settings = {"activation": "relu"}
# decoder_struct_param = [
#     [32, "ConvTranspose2d", {"kernel_size": 3, "stride": 2}],
#     [32, "ConvTranspose2d", {"kernel_size": 3, "stride": 2}],
#     [1, "ConvTranspose2d", {"kernel_size": 3, "stride": 2}],
# ]

# conv_encoder = Conv_Autoencoder(
#     encoder_struct_param,
#     decoder_struct_param,
#     enc_flatten_size = 512,
#     latent_size = 2,
#     settings = {"activation": "leakyReluFlat"},
#     is_cuda = True,
# )


# In[4]:


# patience = 30
# epochs = 1000
# batch_size = 128
# lr = 1e-3
# reg_amp = 1e-7

# (X_train_all, y_train_all), (X_test_all, y_test_all) = combine_dataset(tasks_train)
# optimizer = optim.Adam(conv_encoder.parameters(), lr = lr)
# train_loader = data_utils.DataLoader(X_train_all.data, batch_size = batch_size, shuffle = True)
# early_stopping = Early_Stopping(patience = patience)
# to_stop = False

# for epoch in range(epochs):
#     to_stop = train_epoch_pretrain(train_loader, X_test_all, conv_encoder)
#     if to_stop:
#         print("Early stopping at iteration {0}".format(i))
#         break


# ## Training conv_encoder and prediction at the same time:

# In[5]:


# task = tasks_train[list(tasks_train.keys())[0]]
# ((X_motion_train, y_motion_train), (X_motion_test, y_motion_test)), _ = task


# In[6]:


# encoder_struct_param = [
#     [32, "Conv2d", {"kernel_size": 3, "stride": 2}],
#     [32, "Conv2d", {"kernel_size": 3, "stride": 2}],
#     [32, "Conv2d", {"kernel_size": 3, "stride": 2}],
# ]
# settings = settings = {"activation": "relu"}
# decoder_struct_param = [
#     [32, "ConvTranspose2d", {"kernel_size": 3, "stride": 2}],
#     [32, "ConvTranspose2d", {"kernel_size": 3, "stride": 2}],
#     [1, "ConvTranspose2d", {"kernel_size": 3, "stride": 2}],
# ]

# conv_encoder = Conv_Autoencoder(
#     encoder_struct_param,
#     decoder_struct_param,
#     enc_flatten_size = 512,
#     latent_size = 2,
#     settings = {"activation": "leakyReluFlat"},
#     is_cuda = True,
# )

# struct_param_predictor = [[40, "Simple_Layer", {}], [40, "Simple_Layer", {}], [(1, 2), "Simple_Layer", {"activation": "linear"}]]
# predictor = Net(input_size = (3, 2), struct_param = struct_param_predictor, settings = {"activation": "leakyReluFlat"}, is_cuda = is_cuda)


# In[7]:


# forward_steps = [1,2,4]
# aux_coeff = 0.3
# # With 10 neurons in the predictor:
# # Joint training:
# batch_size = 1000
# epochs = 1000
# patience = 100
# lr = 1e-3
# reg_amp_conv = 1e-6
# reg_amp_predictor = 1e-5
# reg_multiplier = np.linspace(0, 1, epochs + 1) ** 2
# dataset_motion_train = data_utils.TensorDataset(X_motion_train.data, y_motion_train.data)
# motion_train_loader = data_utils.DataLoader(dataset_motion_train, batch_size = batch_size, shuffle = True)

# optimizer_motion = optim.Adam(params = itertools.chain(conv_encoder.parameters(), predictor.parameters()), lr = lr)
# early_stopping_motion = Early_Stopping(patience = patience)
# for epoch in range(epochs):
#     to_stop = train_epoch_joint(motion_train_loader, X_motion_test, y_motion_test, conv_encoder, predictor, forward_steps)
#     if to_stop:
#         print("Early stopping at epoch {0}".format(epoch))
#         break


# ## Setting up:

# In[ ]:


task_id_list = [
# "latent-linear",
# "polynomial-3",
# "Legendre-3",
# "M-sawtooth",
# "M-sin",
# "M-Gaussian",
# "M-tanh",
# "M-softplus",
# "C-sin",
# "C-tanh",
# "bounce-states",
"bounce-images",
]

exp_id = "C-May13"
exp_mode = "meta"
# exp_mode = "finetune"
# exp_mode = "oracle"
is_VAE = False
is_uncertainty_net = False
is_regulated_net = False
is_load_data = False
VAE_beta = 0.2
task_id_list = get_args(task_id_list, 3, type = "tuple")
if task_id_list[0] in ["C-sin", "C-tanh"]:
    statistics_output_neurons = 2 if task_id_list[0] == "C-sin" else 4
    z_size = 2 if task_id_list[0] == "C-sin" else 4
    num_shots = 10
    input_size = 1
    output_size = 1
    reg_amp = 1e-6
    forward_steps = [1]
    is_time_series = False
elif task_id_list[0] == "bounce-states":
    statistics_output_neurons = 8
    num_shots = 100
    z_size = 8
    input_size = 6
    output_size = 2
    reg_amp = 1e-8
    forward_steps = [1]
    is_time_series = True
elif task_id_list[0] == "bounce-images":
    statistics_output_neurons = 8
    num_shots = 100
    z_size = 8
    input_size = 6
    output_size = 2
    reg_amp = 1e-7
    forward_steps = [1]
    is_time_series = True
else:
    raise

is_autoencoder = True
max_forward_steps = 10

lr = 5e-5
num_train_tasks = 100
num_test_tasks = 100
batch_size_task = num_train_tasks
num_iter = 10000
pre_pooling_neurons = 200
num_context_neurons = 0
statistics_pooling = "max"
struct_param_pre_neurons = (60,3)
struct_param_gen_base_neurons = (60,3)
main_hidden_neurons = (40, 40)
activation_gen = "leakyRelu"
activation_model = "leakyRelu"
optim_mode = "indi"
loss_core = "mse"
patience = 300
array_id = 0

exp_id = get_args(exp_id, 1)
exp_mode = get_args(exp_mode, 2)
statistics_output_neurons = get_args(statistics_output_neurons, 4, type = "int")
is_VAE = get_args(is_VAE, 5, type = "bool")
VAE_beta = get_args(VAE_beta, 6, type = "float")
lr = get_args(lr, 7, type = "float")
pre_pooling_neurons = get_args(pre_pooling_neurons, 8, type = "int")
num_context_neurons = get_args(num_context_neurons, 9, type = "int")
statistics_pooling = get_args(statistics_pooling, 10)
struct_param_pre_neurons = get_args(struct_param_pre_neurons, 11, "tuple")
struct_param_gen_base_neurons = get_args(struct_param_gen_base_neurons, 12, "tuple")
main_hidden_neurons = get_args(main_hidden_neurons, 13, "tuple")
reg_amp = get_args(reg_amp, 14, type = "float")
activation_gen = get_args(activation_gen, 15)
activation_model = get_args(activation_model, 16)
optim_mode = get_args(optim_mode, 17)
is_uncertainty_net = get_args(is_uncertainty_net, 18, "bool")
loss_core = get_args(loss_core, 19)
patience = get_args(patience, 20, "int")
forward_steps = get_args(forward_steps, 21, "tuple")
array_id = get_args(array_id, 22)

# Settings:
task_settings = {
    "xlim": (-5, 5),
    "num_examples": num_shots * 2,
    "test_size": 0.5,
}
isParallel = False
inspect_interval = 5
save_interval = 100
num_backwards = 1
is_oracle = (exp_mode == "oracle")
if is_oracle:
    input_size += z_size
print("exp_mode: {0}".format(exp_mode))

# Obtain tasks:
assert len(task_id_list) == 1
dataset_filename = dataset_PATH + task_id_list[0] + "_{0}-shot.p".format(num_shots)
tasks = pickle.load(open(dataset_filename, "rb"))
tasks_train = get_torch_tasks(tasks["tasks_train"], task_id_list[0], num_forward_steps = forward_steps[-1], is_oracle = is_oracle, is_cuda = is_cuda)
tasks_test = get_torch_tasks(tasks["tasks_test"], task_id_list[0], start_id = num_train_tasks, num_tasks = num_test_tasks, num_forward_steps = forward_steps[-1], is_oracle = is_oracle, is_cuda = is_cuda)

# Obtain autoencoder:
aux_coeff = 0.3
if is_autoencoder:
    encoder_struct_param = [
        [32, "Conv2d", {"kernel_size": 3, "stride": 2}],
        [32, "Conv2d", {"kernel_size": 3, "stride": 2}],
        [32, "Conv2d", {"kernel_size": 3, "stride": 2}],
    ]
    settings = settings = {"activation": "relu"}
    decoder_struct_param = [
        [32, "ConvTranspose2d", {"kernel_size": 3, "stride": 2}],
        [32, "ConvTranspose2d", {"kernel_size": 3, "stride": 2}],
        [1, "ConvTranspose2d", {"kernel_size": 3, "stride": 2}],
    ]

    autoencoder = Conv_Autoencoder(
        encoder_struct_param,
        decoder_struct_param,
        enc_flatten_size = 512,
        latent_size = 2,
        settings = {"activation": "leakyReluFlat"},
        is_cuda = is_cuda,
    )
else:
    autoencoder = None
        
        
# Obtain nets:
all_keys = list(tasks_train.keys()) + list(tasks_test.keys())
data_record = {"loss": {key: [] for key in all_keys}, "loss_sampled": {key: [] for key in all_keys}, "mse": {key: [] for key in all_keys},
               "reg": {key: [] for key in all_keys}, "KLD": {key: [] for key in all_keys}}
reg_multiplier = np.linspace(0, 1, num_iter + 1) ** 2
if exp_mode in ["meta"]:
    struct_param_pre = [[struct_param_pre_neurons[0], "Simple_Layer", {}] for _ in range(struct_param_pre_neurons[1])]
    struct_param_pre.append([pre_pooling_neurons, "Simple_Layer", {"activation": "linear"}])
    struct_param_post = None
    struct_param_gen_base = [[struct_param_gen_base_neurons[0], "Simple_Layer", {}] for _ in range(struct_param_gen_base_neurons[1])]
    statistics_Net, generative_Net, generative_Net_logstd = get_nets(input_size = input_size, output_size = output_size, 
                                                                      target_size = len(forward_steps) * output_size, main_hidden_neurons = main_hidden_neurons,
                                                                      pre_pooling_neurons = pre_pooling_neurons, statistics_output_neurons = statistics_output_neurons, num_context_neurons = num_context_neurons,
                                                                      struct_param_pre = struct_param_pre,
                                                                      struct_param_gen_base = struct_param_gen_base,
                                                                      activation_statistics = activation_gen,
                                                                      activation_generative = activation_gen,
                                                                      activation_model = activation_model,
                                                                      statistics_pooling = statistics_pooling,
                                                                      isParallel = isParallel,
                                                                      is_VAE = is_VAE,
                                                                      is_uncertainty_net = is_uncertainty_net,
                                                                      is_cuda = is_cuda,
                                                                     )
    if is_regulated_net:
        struct_param_regulated_Net = [[num_neurons, "Simple_Layer", {}] for num_neurons in main_hidden_neurons]
        struct_param_regulated_Net.append([1, "Simple_Layer", {"activation": "linear"}])
        generative_Net = Net(input_size = input_size, struct_param = struct_param_regulated_Net, settings = {"activation": activation_model})
    master_model = Master_Model(statistics_Net, generative_Net, generative_Net_logstd, is_cuda = is_cuda)
    
    all_parameter_list = [statistics_Net.parameters(), generative_Net.parameters()]
    if is_uncertainty_net:
        all_parameter_list.append(generative_Net_logstd.parameters())
    if is_autoencoder:
        all_parameter_list.append(autoencoder.parameters())
    optimizer = optim.Adam(chain.from_iterable(all_parameter_list), lr = lr)
    reg_dict = {"statistics_Net": {"weight": reg_amp, "bias": reg_amp},
                "generative_Net": {"weight": reg_amp, "bias": reg_amp, "W_gen": reg_amp, "b_gen": reg_amp},
                "autoencoder": {"weight": 1e-7, "bias": 1e-7},
               }
    record_data(data_record, [struct_param_gen_base, struct_param_pre, struct_param_post], ["struct_param_gen_base", "struct_param_pre", "struct_param_post"])
    model = None

elif exp_mode in ["finetune", "oracle"]:
    struct_param_net = [[num_neurons, "Simple_Layer", {}] for num_neurons in main_hidden_neurons]
    struct_param_net.append([output_size, "Simple_Layer", {"activation": "linear"}])
    record_data(data_record, [struct_param_net], ["struct_param_net"])
    model = Net(input_size = input_size,
                  struct_param = struct_param_net,
                  settings = {"activation": activation_model},
                  is_cuda = is_cuda,
                 )
    reg_dict = {"net": {"weight": reg_amp, "bias": reg_amp},
                "autoencoder": {"weight": 1e-7, "bias": 1e-7},
               }
    all_parameter_list = [model.parameters()]
    if is_autoencoder:
        all_parameter_list.append(autoencoder.parameters())
    optimizer = optim.Adam(chain.from_iterable(all_parameter_list), lr = lr)
    statistics_Net = None
    generative_Net = None
    generative_Net_logstd = None
    master_model = None

# Setting up optimizer and loss functions:
forward_steps_idx = torch.LongTensor(np.array(forward_steps) - 1)
if is_cuda:
    forward_steps_idx = forward_steps_idx.cuda()
if loss_core == "mse":
    loss_fun_core = nn.MSELoss(size_average = True)
elif loss_core == "huber":
    loss_fun_core = nn.SmoothL1Loss(size_average = True) 
else:
    raise
if is_VAE:
    criterion = VAE_Loss(criterion = loss_fun_core, prior = "Gaussian", beta = VAE_beta)
else:
    if is_uncertainty_net:
        criterion = Loss_with_uncertainty(core = loss_core)
    else:
        if is_autoencoder:
            criterion = Loss_with_autoencoder(core = loss_core, forward_steps = forward_steps, aux_coeff = aux_coeff, is_cuda = is_cuda)
        else:
            criterion = loss_fun_core
early_stopping = Early_Stopping(patience = patience)

filename = variational_model_PATH + "/trained_models/{0}/ENet_{1}_{2}_input_{3}_({4},{5})_stat_{6}_pre_{7}_pool_{8}_context_{9}_hid_{10}_{11}_{12}_VAE_{13}_{14}_uncer_{15}_lr_{16}_reg_{17}_actgen_{18}_actmodel_{19}_{20}_core_{21}_pat_{22}_for_{23}_{24}_".format(
    exp_id, exp_mode, task_id_list, input_size, num_train_tasks, num_test_tasks, statistics_output_neurons, pre_pooling_neurons, statistics_pooling, num_context_neurons, main_hidden_neurons, struct_param_pre_neurons, struct_param_gen_base_neurons, is_VAE, VAE_beta, is_uncertainty_net, lr, reg_amp, activation_gen, activation_model, optim_mode, loss_core, patience, forward_steps[-1], exp_id)
make_dir(filename)
print(filename)

# Setting up recordings:
info_dict = {"array_id": array_id}
info_dict["data_record"] = data_record
info_dict["model_dict"] = []
record_data(data_record, [exp_id, tasks_train, tasks_test, task_id_list, task_settings, reg_dict, is_uncertainty_net, lr, pre_pooling_neurons, num_backwards, batch_size_task, 
                          statistics_pooling, activation_gen, activation_model], 
            ["exp_id", "tasks_train", "tasks_test", "task_id_list", "task_settings", "reg_dict", "is_uncertainty_net", "lr", "pre_pooling_neurons", "num_backwards", "batch_size_task",
             "statistics_pooling", "activation_gen", "activation_model"])

filename = variational_model_PATH + "/trained_models/{0}/Net_{1}_{2}_input_{3}_({4},{5})_stat_{6}_pre_{7}_pool_{8}_context_{9}_hid_{10}_{11}_{12}_VAE_{13}_{14}_uncer_{15}_lr_{16}_reg_{17}_actgen_{18}_actmodel_{19}_{20}_core_{21}_pat_{22}_{23}_".format(
    exp_id, exp_mode, task_id_list, input_size, num_train_tasks, num_test_tasks, statistics_output_neurons, pre_pooling_neurons, statistics_pooling, num_context_neurons, main_hidden_neurons, struct_param_pre_neurons, struct_param_gen_base_neurons, is_VAE, VAE_beta, is_uncertainty_net, lr, reg_amp, activation_gen, activation_model, optim_mode, loss_core, patience, exp_id)
make_dir(filename)
print(filename)


# ## Pre-train the autoencoder for a few epochs:

# In[ ]:


patience_pre = 30
batch_size = 128
lr_pre = 1e-3
reg_amp_autoencoder = 1e-7
reg_amp_latent = 1e-2

(X_train_all, y_train_all), (X_test_all, y_test_all) = combine_dataset(tasks_train, num = 50)
optimizer_pre = optim.Adam(autoencoder.parameters(), lr = lr_pre)
train_loader = data_utils.DataLoader(X_train_all.data, batch_size = batch_size, shuffle = True)
early_stopping_pre = Early_Stopping(patience = patience_pre)
to_stop = False

print("Pre-train autoencoder:")
for epoch in range(11):
    to_stop = train_epoch_pretrain(train_loader, X_test_all, autoencoder, optimizer_pre, isplot = isplot)
    if to_stop:
        print("Early stopping at iteration {0}".format(i))
        break
print("Pretrain completed!")


# ## Training:

# In[ ]:


# Training:
for i in range(num_iter + 1):
    """Training the meta-autoencoder"""
    chosen_task_keys = np.random.choice(list(tasks_train.keys()), batch_size_task, replace = False).tolist()
    if optim_mode == "indi":
        if is_VAE:
            KLD_total = Variable(torch.FloatTensor([0]), requires_grad = False)
            if is_cuda:
                KLD_total = KLD_total.cuda()
        for task_key, task in tasks_train.items():
            if task_key not in chosen_task_keys:
                continue
            if is_autoencoder:
                ((X_train_obs, y_train_obs), (X_test_obs, y_test_obs)), _ = task
                X_train = forward(autoencoder.encode, X_train_obs)
                y_train = forward(autoencoder.encode, y_train_obs[:, forward_steps_idx])
                X_test = forward(autoencoder.encode, X_test_obs)
                y_test = forward(autoencoder.encode, y_test_obs[:, forward_steps_idx])
            else:
                ((X_train, y_train), (X_test, y_test)), _ = task
            for k in range(num_backwards):
                optimizer.zero_grad()
                if master_model is not None:
                    results = master_model.get_predictions(X_test = X_test, X_train = X_train, y_train = y_train, is_time_series = is_time_series, 
                                                           is_VAE = is_VAE, is_uncertainty_net = is_uncertainty_net, is_regulated_net = is_regulated_net, forward_steps = forward_steps)
                else:
                    results = {}
                    results["y_pred"] = get_forward_pred(model, X_test, forward_steps, is_time_series = is_time_series, jump_step = 2, is_flatten = True)
                if is_VAE:
                    statistics_mu, statistics_logvar = statistics_Net(torch.cat([X_train, y_train], 1))
                    statistics = sample_Gaussian(statistics_mu, statistics_logvar)
                    if is_regulated_net:
                        statistics = get_regulated_statistics(generative_Net, statistics)
                    y_pred = generative_Net(X_test, statistics)
                    loss, KLD = criterion(y_pred, y_test, mu = statistics_mu, logvar = statistics_logvar)
                    KLD_total = KLD_total + KLD
                else:
                    if is_uncertainty_net:
                        statistics_mu, statistics_logvar = statistics_Net(torch.cat([X_train, y_train], 1))
                        y_pred = generative_Net(X_test, statistics_mu)
                        y_pred_logstd = generative_Net_logstd(X_test, statistics_logvar)
                        loss = criterion(y_pred, y_test, log_std = y_pred_logstd)
                    else:
                        if is_autoencoder:
                            loss = criterion(X_test, results["y_pred"], X_test_obs, y_test_obs, autoencoder, verbose = False)
                        else:
                            loss = criterion(results["y_pred"], y_test)
                reg = get_reg(reg_dict, statistics_Net = statistics_Net, generative_Net = generative_Net, net = model, autoencoder = autoencoder, is_cuda = is_cuda)
                loss = loss + reg * reg_multiplier[i]
                loss.backward(retain_graph = True)
                optimizer.step()
        # Perform gradient on the KL-divergence:
        if is_VAE:
            KLD_total = KLD_total / batch_size_task
            optimizer.zero_grad()
            KLD_total.backward()
            optimizer.step()
            record_data(data_record, [KLD_total], ["KLD_total"])
    elif optim_mode == "sum":
        optimizer.zero_grad()
        loss_total = Variable(torch.FloatTensor([0]), requires_grad = False)
        if is_cuda:
            loss_total = loss_total.cuda()
        for task_key, task in tasks_train.items():
            if task_key not in chosen_task_keys:
                continue
            if is_autoencoder:
                ((X_train_obs, y_train_obs), (X_test_obs, y_test_obs)), _ = task
                X_train = forward(autoencoder.encode, X_train_obs)
                y_train = forward(autoencoder.encode, y_train_obs)
                X_test = forward(autoencoder.encode, X_test_obs)
                y_test = forward(autoencoder.encode, y_test_obs)
            else:
                ((X_train, y_train), (X_test, y_test)), _ = task
            if master_model is not None:
                results = master_model.get_predictions(X_test = X_test, X_train = X_train, y_train = y_train, is_time_series = is_time_series, 
                                                       is_VAE = is_VAE, is_uncertainty_net = is_uncertainty_net, is_regulated_net = is_regulated_net, forward_steps = forward_steps)
            else:
                results = {}
                results["y_pred"] = get_forward_pred(model, X_test, forward_steps, is_time_series = is_time_series, jump_step = 2, is_flatten = True)
            if is_VAE:
                statistics_mu, statistics_logvar = statistics_Net(torch.cat([X_train, y_train], 1))
                statistics = sample_Gaussian(statistics_mu, statistics_logvar)
                y_pred = generative_Net(X_test, statistics)
                loss, KLD = criterion(y_pred, y_test, mu = statistics_mu, logvar = statistics_logvar)
                loss = loss + KLD
            else:
                if is_uncertainty_net:
                    statistics_mu, statistics_logvar = statistics_Net(torch.cat([X_train, y_train], 1))
                    y_pred = generative_Net(X_test, statistics_mu)
                    y_pred_logstd = generative_Net_logstd(X_test, statistics_logvar)
                    loss = criterion(y_pred, y_test, log_std = y_pred_logstd)
                else:
                    if is_autoencoder:
                        loss = criterion(X_test, results["y_pred"], X_test_obs, y_test_obs, autoencoder, verbose = False)
                    else:
                        loss = criterion(results["y_pred"], y_test)
            reg = get_reg(reg_dict, statistics_Net = statistics_Net, generative_Net = generative_Net, autoencoder = autoencoder, net = model, is_cuda = is_cuda)
            loss_total = loss_total + loss + reg * reg_multiplier[i]
        loss_total.backward()
        optimizer.step()
    else:
        raise Exception("optim_mode {0} not recognized!".format(optim_mode))    

    loss_test_record = []
    for task_key, task in tasks_test.items():
        loss_test, _, _, _ = evaluate(task, master_model = master_model, model = model, criterion = criterion, is_VAE = is_VAE, is_regulated_net = is_regulated_net, autoencoder = autoencoder, forward_steps = forward_steps)
        loss_test_record.append(loss_test)
    to_stop = early_stopping.monitor(np.mean(loss_test_record))

    # Validation and visualization:
    if i % inspect_interval == 0 or to_stop:
        print("=" * 50)
        print("training tasks:")
        for task_key, task in tasks_train.items():
            loss_test, loss_test_sampled, mse, KLD_test = evaluate(task, master_model = master_model, model = model, criterion = criterion, is_VAE = is_VAE, is_regulated_net = is_regulated_net, autoencoder = autoencoder, forward_steps = forward_steps)
            reg = get_reg(reg_dict, statistics_Net = statistics_Net, generative_Net = generative_Net, autoencoder = autoencoder, net = model, is_cuda = is_cuda).data[0] * reg_multiplier[i]
            data_record["loss"][task_key].append(loss_test)
            data_record["loss_sampled"][task_key].append(loss_test_sampled)
            data_record["mse"][task_key].append(mse)
            data_record["reg"][task_key].append(reg)
            data_record["KLD"][task_key].append(KLD_test)
            print('{0}\ttrain\t{1}  \tloss: {2:.5f}\tloss_sampled:{3:.5f} \tmse:{4:.5f}\tKLD:{5:.6f}\treg:{6:.6f}'.format(i, task_key, loss_test, loss_test_sampled, mse, KLD_test, reg))
        for task_key, task in tasks_test.items():
            loss_test, loss_test_sampled, mse, KLD_test = evaluate(task, master_model = master_model, model = model, criterion = criterion, is_VAE = is_VAE, is_regulated_net = is_regulated_net, autoencoder = autoencoder, forward_steps = forward_steps)
            reg = get_reg(reg_dict, statistics_Net = statistics_Net, generative_Net = generative_Net, autoencoder = autoencoder, net = model, is_cuda = is_cuda).data[0] * reg_multiplier[i]
            data_record["loss"][task_key].append(loss_test)
            data_record["loss_sampled"][task_key].append(loss_test_sampled)
            data_record["mse"][task_key].append(mse)
            data_record["reg"][task_key].append(reg)
            data_record["KLD"][task_key].append(KLD_test)
            print('{0}\ttest\t{1}  \tloss: {2:.5f}\tloss_sampled:{3:.5f} \tmse:{4:.5f}\tKLD:{5:.6f}\treg:{6:.6f}'.format(i, task_key, loss_test, loss_test_sampled, mse, KLD_test, reg))
        loss_train_list = [data_record["loss"][task_key][-1] for task_key in tasks_train]
        loss_test_list = [data_record["loss"][task_key][-1] for task_key in tasks_test]
        loss_train_sampled_list = [data_record["loss_sampled"][task_key][-1] for task_key in tasks_train]
        loss_test_sampled_list = [data_record["loss_sampled"][task_key][-1] for task_key in tasks_test]
        mse_train_list = [data_record["mse"][task_key][-1] for task_key in tasks_train]
        mse_test_list = [data_record["mse"][task_key][-1] for task_key in tasks_test]
        reg_train_list = [data_record["reg"][task_key][-1] for task_key in tasks_train]
        reg_test_list = [data_record["reg"][task_key][-1] for task_key in tasks_test]
        mse_few_shot = plot_few_shot_loss(master_model, tasks_test, isplot = isplot, autoencoder = autoencoder, forward_steps = forward_steps, criterion = criterion)
        record_data(data_record, 
                    [np.mean(loss_train_list), np.median(loss_train_list), np.mean(reg_train_list), i,
                     np.mean(loss_test_list), np.median(loss_test_list), np.mean(reg_test_list),
                     np.mean(loss_train_sampled_list), np.median(loss_train_sampled_list), 
                     np.mean(loss_test_sampled_list), np.median(loss_test_sampled_list),
                     np.mean(mse_train_list), np.median(mse_train_list), 
                     np.mean(mse_test_list), np.median(mse_test_list), 
                     mse_few_shot,
                    ], 
                    ["loss_mean_train", "loss_median_train", "reg_mean_train", "iter",
                     "loss_mean_test", "loss_median_test", "reg_mean_test",
                     "loss_sampled_mean_train", "loss_sampled_median_train",
                     "loss_sampled_mean_test", "loss_sampled_median_test", 
                     "mse_mean_train", "mse_median_train", "mse_mean_test", "mse_median_test", 
                     "mse_few_shot",
                    ])
        if isplot:
            plot_data_record(data_record, idx = -1, is_VAE = is_VAE)
        print("Summary:")
        print('\n{0}\ttrain\tloss_mean: {1:.5f}\tloss_median: {2:.5f}\tmse_mean: {3:.6f}\tmse_median: {4:.6f}\treg: {5:.6f}'.format(i, data_record["loss_mean_train"][-1], data_record["loss_median_train"][-1], data_record["mse_mean_train"][-1], data_record["mse_median_train"][-1], data_record["reg_mean_train"][-1]))
        print('{0}\ttest\tloss_mean: {1:.5f}\tloss_median: {2:.5f}\tmse_mean: {3:.6f}\tmse_median: {4:.6f}\treg: {5:.6f}'.format(i, data_record["loss_mean_test"][-1], data_record["loss_median_test"][-1], data_record["mse_mean_test"][-1], data_record["mse_median_test"][-1], data_record["reg_mean_test"][-1]))
        if is_VAE and "KLD_total" in locals():
            print("KLD_total: {0:.5f}".format(KLD_total.data[0]))
        if isplot:
            plot_data_record(data_record, is_VAE = is_VAE)

        # Plotting y_pred vs. y_target:
        statistics_list_train, z_list_train = plot_task_ensembles(tasks_train, master_model = master_model, model = model, is_VAE = is_VAE, is_regulated_net = is_regulated_net, autoencoder = autoencoder, title = "y_pred_train vs. y_train", isplot = isplot, forward_steps = forward_steps, )
        statistics_list_test, z_list_test = plot_task_ensembles(tasks_test, master_model = master_model, model = model, is_VAE = is_VAE, is_regulated_net = is_regulated_net, autoencoder = autoencoder, title = "y_pred_test vs. y_test", isplot = isplot, forward_steps = forward_steps, )
        record_data(data_record, [np.array(z_list_train), np.array(z_list_test), np.array(statistics_list_train), np.array(statistics_list_test)], 
                    ["z_list_train_list", "z_list_test_list", "statistics_list_train_list", "statistics_list_test_list"])
        if isplot:
            print("train statistics vs. z:")
            plot_statistics_vs_z(z_list_train, statistics_list_train)
            print("test statistics vs. z:")
            plot_statistics_vs_z(z_list_test, statistics_list_test)

            # Plotting individual test data:
            if "bounce" in task_id_list[0]:
                if "bounce-images" in task_id_list[0]:
                    plot_tasks(tasks_test, master_model = master_model, model = model, autoencoder = autoencoder, forward_steps = forward_steps, num_tasks = min(3, num_test_tasks))
                plot_individual_tasks_bounce(tasks_test, num_examples_show = 40, num_tasks_show = 6, master_model = master_model, model = model, autoencoder = autoencoder, num_shots = 200, target_forward_steps = len(forward_steps), eval_forward_steps = len(forward_steps))
            else:
                print("train tasks:")
                plot_individual_tasks(tasks_train, master_model = master_model, model = model, is_VAE = is_VAE, is_regulated_net = is_regulated_net, xlim = task_settings["xlim"])
                print("test tasks:")
                plot_individual_tasks(tasks_test, master_model = master_model, model = model, is_VAE = is_VAE, is_regulated_net = is_regulated_net, xlim = task_settings["xlim"])
        print("=" * 50 + "\n\n")
        try:
            sys.stdout.flush()
        except:
            pass
    if i % save_interval == 0 or to_stop:
        model_save = master_model if master_model is not None else model
        record_data(info_dict, [model_save.model_dict], ["model_dict"])
        make_dir(filename[:-1] + "/conv-meta_master-model")
        torch.save(autoencoder.state_dict(), filename[:-1] + "/conv-meta_autoencoder_{0}.p".format(i))
    if to_stop:
        print("The training loss stops decreasing for {0} steps. Early stopping at {1}.".format(patience, i))
        break


# Plotting:
if isplot:
    for task_key in tasks_train:
        plt.semilogy(data_record["loss"][task_key], alpha = 0.6)
    plt.show()
    for task_key in tasks_test:
        plt.semilogy(data_record["loss"][task_key], alpha = 0.6)
    plt.show()
print("completed")
sys.stdout.flush()
pickle.dump(info_dict, open(filename + "info.p", "wb"))

