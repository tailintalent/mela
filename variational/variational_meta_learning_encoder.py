
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
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from AI_scientist.util import plot_matrices, make_dir, get_struct_str, get_args, Early_Stopping, record_data, manifold_embedding
from AI_scientist.settings.filepath import variational_model_PATH
from AI_scientist.pytorch.modules import Simple_Layer
from AI_scientist.pytorch.net import Net, ConvNet
from AI_scientist.pytorch.util_pytorch import Loss_with_uncertainty, get_criterion
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


# In[2]:


def combine_dataset(tasks):
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    for task in tasks.values():
        ((X_train, y_train), (X_test, y_test)), info = task
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_test_list.append(X_test)
        y_test_list.append(y_test)
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


def train_epoch_pretrain(train_loader, X_test_all, conv_encoder):
    for batch_id, X_batch in enumerate(train_loader):
        X_batch = Variable(X_batch)
        optimizer.zero_grad()
        reconstruct = forward(conv_encoder, X_batch)
        reg = conv_encoder.get_regularization(source = ["weight", "bias"]) * reg_amp
        loss_train = nn.MSELoss()(reconstruct, X_batch)
        loss_train.backward()
        optimizer.step()

    reconstruct_test = forward(conv_encoder, X_test_all)
    loss_test = nn.MSELoss()(reconstruct_test, X_test_all)
    print("epoch {0} \tloss_train: {1:.6f}\tloss_test: {2:.6f}\treg: {3:.6f}".format(epoch, loss_train.data[0], loss_test.data[0], reg.data[0]))
    if epoch % 10 == 0:
        plot_matrices(X_batch[0].cpu().data.numpy(), images_per_row = 5)
        latent = forward(conv_encoder.encode, X_batch)
        print("latent: {0}".format(latent.cpu().data.numpy()[0]))
        plot_matrices(reconstruct[0].cpu().data.numpy(), images_per_row = 5)
        plot_encoding(X_train_all[:,:1], conv_encoder, "axis")
    return to_stop


def train_epoch_joint(motion_train_loader, X_motion_test, y_motion_test, conv_encoder, predictor, forward_steps, aux_coeff):
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
    if epoch % 10 == 0:
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
#     to_stop = train_epoch(train_loader, X_test_all, conv_encoder)
#     if to_stop:
#         print("Early stopping at iteration {0}".format(i))
#         break


# ## Training conv_encoder and prediction at the same time:

# In[5]:


# task = tasks_train(list(tasks_train.keys())[0])
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
# "bounce-states",
"bounce-images",
]
exp_id = "test"
input_size = 6
statistics_output_neurons = 8
is_VAE = False
is_uncertainty_net = False
is_regulated_net = False
is_load_data = False
is_autoencoder = True
forward_steps = [1]
VAE_beta = 0.2

output_size = 2
lr = 5e-5
num_train_tasks = 50
num_test_tasks = 20
batch_size_task = min(100, num_train_tasks)
num_backwards = 1
num_iter = 3000
pre_pooling_neurons = 200
num_context_neurons = 0
statistics_pooling = "max"
main_hidden_neurons = (40, 40)
patience = 200
reg_amp = 1e-6
activation_gen = "leakyRelu"
activation_model = "leakyRelu"
optim_mode = "indi"
loss_core = "mse"
array_id = "0"

exp_id = get_args(exp_id, 1)
task_id_list = get_args(task_id_list, 2, type = "tuple")
statistics_output_neurons = get_args(statistics_output_neurons, 3, type = "int")
is_VAE = get_args(is_VAE, 4, type = "bool")
VAE_beta = get_args(VAE_beta, 5, type = "float")
lr = get_args(lr, 6, type = "float")
batch_size_task = get_args(batch_size_task, 7, type = "int")
pre_pooling_neurons = get_args(pre_pooling_neurons, 8, type = "int")
num_context_neurons = get_args(num_context_neurons, 9, type = "int")
statistics_pooling = get_args(statistics_pooling, 10)
main_hidden_neurons = get_args(main_hidden_neurons, 11, "tuple")
reg_amp = get_args(reg_amp, 12, type = "float")
activation_gen = get_args(activation_gen, 13)
activation_model = get_args(activation_model, 14)
optim_mode = get_args(optim_mode, 15)
is_uncertainty_net = get_args(is_uncertainty_net, 16, "bool")
loss_core = get_args(loss_core, 17)
array_id = get_args(array_id, 18)

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    isplot = True
except:
    isplot = False

# Settings:
reg_dict = {"statistics_Net": {"weight": reg_amp, "bias": reg_amp},
            "generative_Net": {"weight": reg_amp, "bias": reg_amp, "W_gen": reg_amp, "b_gen": reg_amp},
            "autoencoder": {"weight": 1e-7, "bias": 1e-7}}
reg_multiplier = np.linspace(0, 1, num_iter + 1) ** 2
task_settings = {
    "zdim": 1,
    "z_settings": ["Gaussian", (0, 1)],
    "num_layers": 1,
    "xlim": (-4, 4),
    "activation": "softplus",
    "input_size": input_size,
    "num_examples": 2000,
}
struct_param_pre = [
        [60, "Simple_Layer", {}],
        [60, "Simple_Layer", {}],
        [60, "Simple_Layer", {}],
        [pre_pooling_neurons, "Simple_Layer", {"activation": "linear"}],
    ]
struct_param_post = None
struct_param_gen_base = [
        [60, "Simple_Layer", {}],
        [60, "Simple_Layer", {}],
        [60, "Simple_Layer", {}],
]
isParallel = False
inspect_interval = 5
save_interval = 50
filename = variational_model_PATH + "/trained_models/{0}/Net_{1}_input_{2}_({3},{4})_stat_{5}_pre_{6}_pool_{7}_context_{8}_hid_{9}_batch_{10}_back_{11}_VAE_{12}_{13}_uncer_{14}_lr_{15}_reg_{16}_actgen_{17}_actmodel_{18}_struct_{19}_{20}_core_{21}_{22}_".format(
    exp_id, task_id_list, input_size, num_train_tasks, num_test_tasks, statistics_output_neurons, pre_pooling_neurons, statistics_pooling, num_context_neurons, main_hidden_neurons, batch_size_task, num_backwards, is_VAE, VAE_beta, is_uncertainty_net, lr, reg_amp, activation_gen, activation_model, get_struct_str(struct_param_gen_base), optim_mode, loss_core, exp_id)
make_dir(filename)
print(filename)

# Obtain nets:
statistics_Net, generative_Net, generative_Net_logstd = get_nets(input_size = input_size, output_size = output_size, main_hidden_neurons = main_hidden_neurons,
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
        is_cuda = True,
    )
else:
    autoencoder = None
forward_steps_idx = torch.LongTensor(np.array(forward_steps) - 1).cuda()

if is_regulated_net:
    struct_param_regulated_Net = [
            [40, "Simple_Layer", {}],
            [40, "Simple_Layer", {}],
            [1, "Simple_Layer", {"activation": "linear"}],
    ]
    generative_Net = Net(input_size = input_size, struct_param = struct_param_regulated_Net, settings = {"activation": activation_model})
master_model = Master_Model(statistics_Net, generative_Net, generative_Net_logstd, is_cuda = is_cuda)

# Setting up optimizer and loss functions:
all_parameter_list = [statistics_Net.parameters(), generative_Net.parameters()]
if is_uncertainty_net:
    all_parameter_list.append(generative_Net_logstd.parameters())
if is_autoencoder:
    all_parameter_list.append(autoencoder.parameters())
optimizer = optim.Adam(chain.from_iterable(all_parameter_list), lr = lr)

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
            criterion = Loss_with_autoencoder(core = loss_core, forward_steps = forward_steps, aux_coeff = 0.5)
        else:
            criterion = loss_fun_core
early_stopping = Early_Stopping(patience = patience)

# Obtain tasks:
if is_load_data:
    try:
        dataset = pickle.load(open(filename + "data.p", "rb"))
        tasks_train = dataset["tasks_train"]
        tasks_test = dataset["tasks_test"]
        print("dataset loaded.")
    except:
        print("dataset do not exist. Create one")
        tasks_train, tasks_test = get_tasks(task_id_list, num_train_tasks, num_test_tasks, task_settings = task_settings, is_cuda = is_cuda, forward_steps = list(range(1,15)))
        dataset = {"tasks_train": tasks_train, "tasks_test": tasks_test}
#         pickle.dump(dataset, open(filename + "data.p", "wb"))
else:
    tasks_train, tasks_test = get_tasks(task_id_list, num_train_tasks, num_test_tasks, task_settings = task_settings, is_cuda = is_cuda, forward_steps = list(range(1,15)))
    dataset = {"tasks_train": tasks_train, "tasks_test": tasks_test}
#     pickle.dump(dataset, open(filename + "data.p", "wb"))
    print("dataset saved.")


# Setting up recordings:
all_keys = list(tasks_train.keys()) + list(tasks_test.keys())
data_record = {"loss": {key: [] for key in all_keys}, "loss_sampled": {key: [] for key in all_keys}, "mse": {key: [] for key in all_keys},
               "reg": {key: [] for key in all_keys}, "KLD": {key: [] for key in all_keys}}
info_dict = {"array_id": array_id}
info_dict["data_record"] = data_record
info_dict["model_dict"] = []
record_data(data_record, [exp_id, tasks_train, tasks_test, task_id_list, task_settings, reg_dict, is_uncertainty_net, lr, pre_pooling_neurons, num_backwards, batch_size_task, 
                          struct_param_gen_base, struct_param_pre, struct_param_post, statistics_pooling, activation_gen, activation_model], 
            ["exp_id", "tasks_train", "tasks_test", "task_id_list", "task_settings", "reg_dict", "is_uncertainty_net", "lr", "pre_pooling_neurons", "num_backwards", "batch_size_task",
             "struct_param_gen_base", "struct_param_pre", "struct_param_post", "statistics_pooling", "activation_gen", "activation_model"])


# In[6]:


def train(statistics_Net, generative_Net, autoencoder, tasks_train, tasks_test, epoch):
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
                if is_VAE:
                    statistics_mu, statistics_logvar = statistics_Net(torch.cat([X_train, y_train], 1))
                    statistics = sample_Gaussian(statistics_mu, statistics_logvar)
                    if is_regulated_net:
                        statistics = get_regulated_statistics(generative_Net, statistics)
                    y_pred = generative_Net(X_train, statistics)
                    loss, KLD = criterion(y_pred, y_train, mu = statistics_mu, logvar = statistics_logvar)
                    KLD_total = KLD_total + KLD
                else:
                    if is_uncertainty_net:
                        statistics_mu, statistics_logvar = statistics_Net(torch.cat([X_train, y_train], 1))
                        y_pred = generative_Net(X_train, statistics_mu)
                        y_pred_logstd = generative_Net_logstd(X_train, statistics_logvar)
                        loss = criterion(y_pred, y_train, log_std = y_pred_logstd)
                    else:
                        statistics = statistics_Net(torch.cat([X_train, y_train], 1))
                        if is_regulated_net:
                            statistics = get_regulated_statistics(generative_Net, statistics)
                        if is_autoencoder:
                            generative_Net.set_latent_param(statistics)
                            y_pred = get_forward_pred(generative_Net, X_train, forward_steps)
                            loss = criterion(X_train, y_pred, X_train_obs, y_train_obs, autoencoder, verbose = True)
                        else:
                            y_pred = generative_Net(X_train, statistics)
                            loss = criterion(y_pred, y_train)
                reg = get_reg(reg_dict, statistics_Net = statistics_Net, generative_Net = generative_Net, autoencoder = autoencoder, is_cuda = is_cuda)
                loss = loss + reg * reg_multiplier[epoch]
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
            if is_VAE:
                statistics_mu, statistics_logvar = statistics_Net(torch.cat([X_train, y_train], 1))
                statistics = sample_Gaussian(statistics_mu, statistics_logvar)
                y_pred = generative_Net(X_train, statistics)
                loss, KLD = criterion(y_pred, y_train, mu = statistics_mu, logvar = statistics_logvar)
                loss = loss + KLD
            else:
                if is_uncertainty_net:
                    statistics_mu, statistics_logvar = statistics_Net(torch.cat([X_train, y_train], 1))
                    y_pred = generative_Net(X_train, statistics_mu)
                    y_pred_logstd = generative_Net_logstd(X_train, statistics_logvar)
                    loss = criterion(y_pred, y_train, log_std = y_pred_logstd)
                else:
                    statistics = statistics_Net(torch.cat([X_train, y_train], 1))
                    if is_regulated_net:
                        statistics = get_regulated_statistics(generative_Net, statistics)
                    if is_autoencoder:
                        generative_Net.set_latent_param(statistics)
                        y_pred = get_forward_pred(generative_Net, X_train, forward_steps)
                        loss = criterion(X_train, y_pred, X_train_obs, y_train_obs, autoencoder)
                    else:
                        y_pred = generative_Net(X_train, statistics)
                        loss = criterion(y_pred, y_train)    
            reg = get_reg(reg_dict, statistics_Net = statistics_Net, generative_Net = generative_Net, autoencoder = autoencoder, is_cuda = is_cuda)
            loss_total = loss_total + loss + reg * reg_multiplier[epoch]
        loss_total.backward()
        optimizer.step()
    else:
        raise Exception("optim_mode {0} not recognized!".format(optim_mode))    

    loss_test_record = []
    for task_key, task in tasks_test.items():
        loss_test, _, _, _ = evaluate(task, statistics_Net, generative_Net, generative_Net_logstd = generative_Net_logstd, criterion = criterion, is_VAE = is_VAE, is_regulated_net = is_regulated_net, autoencoder = autoencoder, forward_steps = forward_steps)
        loss_test_record.append(loss_test)
    to_stop = early_stopping.monitor(np.mean(loss_test_record))
    
    return to_stop


# ## Training:

# In[ ]:


# Training:
for i in range(num_iter + 1):
    to_stop = train(statistics_Net, generative_Net, autoencoder, tasks_train, tasks_test, epoch = i)

    # Validation and visualization:
    if i % inspect_interval == 0 or to_stop:
        print("=" * 50)
        print("training tasks:")
        for task_key, task in tasks_train.items():
            (_, (X_test, y_test)), _ = task
            loss_test, loss_test_sampled, mse, KLD_test = evaluate(task, statistics_Net, generative_Net, generative_Net_logstd = generative_Net_logstd, criterion = criterion, is_VAE = is_VAE, is_regulated_net = is_regulated_net, autoencoder = autoencoder, forward_steps = forward_steps)
            reg = get_reg(reg_dict, statistics_Net = statistics_Net, generative_Net = generative_Net, autoencoder = autoencoder, is_cuda = is_cuda).data[0] * reg_multiplier[i]
            data_record["loss"][task_key].append(loss_test)
            data_record["loss_sampled"][task_key].append(loss_test_sampled)
            data_record["mse"][task_key].append(mse)
            data_record["reg"][task_key].append(reg)
            data_record["KLD"][task_key].append(KLD_test)
            print('{0}\ttrain\t{1}  \tloss: {2:.5f}\tloss_sampled:{3:.5f} \tmse:{4:.5f}\tKLD:{5:.6f}\treg:{6:.6f}'.format(i, task_key, loss_test, loss_test_sampled, mse, KLD_test, reg))
        for task_key, task in tasks_test.items():
            (_, (X_test, y_test)), _ = task
            loss_test, loss_test_sampled, mse, KLD_test = evaluate(task, statistics_Net, generative_Net, generative_Net_logstd = generative_Net_logstd, criterion = criterion, is_VAE = is_VAE, is_regulated_net = is_regulated_net, autoencoder = autoencoder, forward_steps = forward_steps)
            reg = get_reg(reg_dict, statistics_Net = statistics_Net, generative_Net = generative_Net, autoencoder = autoencoder, is_cuda = is_cuda).data[0] * reg_multiplier[i]
            data_record["loss"][task_key].append(loss_test)
            data_record["loss_sampled"][task_key].append(loss_test_sampled)
            data_record["mse"][task_key].append(mse)
            data_record["reg"][task_key].append(reg)
            data_record["KLD"][task_key].append(KLD_test)
            print('{0}\ttrain\t{1}  \tloss: {2:.5f}\tloss_sampled:{3:.5f} \tmse:{4:.5f}\tKLD:{5:.6f}\treg:{6:.6f}'.format(i, task_key, loss_test, loss_test_sampled, mse, KLD_test, reg))
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
        statistics_list_train, z_list_train = plot_task_ensembles(tasks_train, statistics_Net, generative_Net, is_VAE = is_VAE, is_regulated_net = is_regulated_net, autoencoder = autoencoder, title = "y_pred_train vs. y_train", isplot = isplot, forward_steps = forward_steps, )
        statistics_list_test, z_list_test = plot_task_ensembles(tasks_test, statistics_Net, generative_Net, is_VAE = is_VAE, is_regulated_net = is_regulated_net, autoencoder = autoencoder, title = "y_pred_test vs. y_test", isplot = isplot, forward_steps = forward_steps, )
        record_data(data_record, [np.array(z_list_train), np.array(z_list_test), np.array(statistics_list_train), np.array(statistics_list_test)], 
                    ["z_list_train_list", "z_list_test_list", "statistics_list_train_list", "statistics_list_test_list"])
        if isplot:
            print("train statistics vs. z:")
            plot_statistics_vs_z(z_list_train, statistics_list_train)
            print("test statistics vs. z:")
            plot_statistics_vs_z(z_list_test, statistics_list_test)

            # Plotting individual test data:
            if "bounce" in task_id_list[0]:
                plot_individual_tasks_bounce(tasks_test, num_examples_show = 40, num_tasks_show = 6, master_model = master_model, autoencoder = autoencoder, num_shots = 200, forward_steps = forward_steps)
            else:
                print("train tasks:")
                plot_individual_tasks(tasks_train, statistics_Net, generative_Net, generative_Net_logstd = generative_Net_logstd, is_VAE = is_VAE, is_regulated_net = is_regulated_net, xlim = task_settings["xlim"])
                print("test tasks:")
                plot_individual_tasks(tasks_test, statistics_Net, generative_Net, generative_Net_logstd = generative_Net_logstd, is_VAE = is_VAE, is_regulated_net = is_regulated_net, xlim = task_settings["xlim"])
        print("=" * 50 + "\n\n")
        try:
            sys.stdout.flush()
        except:
            pass
    if i % save_interval == 0 or to_stop:
        record_data(info_dict, [master_model.model_dict, i], ["model_dict", "iter"])
        if is_autoencoder:
            record_data(info_dict, [autoencoder.state_dict()], ["autoencoder"])
        pickle.dump(info_dict, open(filename + "info.p", "wb"))
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

