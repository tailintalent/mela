
# coding: utf-8

# In[1]:


import numpy as np
import pickle
from itertools import chain
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from AI_scientist.util import plot_matrices, make_dir, get_struct_str, get_args, Early_Stopping, record_data, manifold_embedding
from AI_scientist.settings.filepath import variational_model_PATH
from AI_scientist.pytorch.net import Net
from AI_scientist.pytorch.util_pytorch import Loss_with_uncertainty
from AI_scientist.variational.variational_meta_learning import Master_Model, Statistics_Net, Generative_Net, load_model_dict, get_regulated_statistics
from AI_scientist.variational.variational_meta_learning import VAE_Loss, sample_Gaussian, clone_net, get_nets, get_tasks, evaluate, get_reg, load_trained_models
from AI_scientist.variational.variational_meta_learning import plot_task_ensembles, plot_individual_tasks, plot_statistics_vs_z, plot_data_record, get_corrcoef
from AI_scientist.variational.variational_meta_learning import plot_few_shot_loss, plot_individual_tasks_bounce
from AI_scientist.variational.variational_meta_learning import get_latent_model_data, get_polynomial_class, get_Legendre_class, get_master_function

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
is_cuda = torch.cuda.is_available()


# ### Training:

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
"bounce-states",
# "bounce-images",
]
exp_id = "comparison"
input_size = 6
statistics_output_neurons = 8
is_VAE = False
is_uncertainty_net = False
is_regulated_net = False
is_load_data = False
VAE_beta = 0.2
exp_mode = "baseline"

output_size = 2
lr = 5e-5
num_train_tasks = 100
num_test_tasks = 50
batch_size_task = min(100, num_train_tasks)
num_backwards = 1
num_iter = 20000
pre_pooling_neurons = 400
num_context_neurons = 0
statistics_pooling = "max"
main_hidden_neurons = (40, 40, 40)
patience = 400
reg_amp = 1e-6
activation_gen = "leakyRelu"
activation_model = "leakyRelu"
optim_mode = "indi"
loss_core = "huber"
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
reg_dict = {"net": {"weight": reg_amp, "bias": reg_amp}}
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
inspect_interval = 50
save_interval = 500
filename = variational_model_PATH + "/trained_models/{0}/Net_{1}_{2}_input_{3}_({4},{5})_stat_{6}_pre_{7}_pool_{8}_context_{9}_hid_{10}_batch_{11}_back_{12}_VAE_{13}_{14}_uncer_{15}_lr_{16}_reg_{17}_actgen_{18}_actmodel_{19}_struct_{20}_{21}_core_{22}_{23}_".format(
    exp_id, exp_mode, task_id_list, input_size, num_train_tasks, num_test_tasks, statistics_output_neurons, pre_pooling_neurons, statistics_pooling, num_context_neurons, main_hidden_neurons, batch_size_task, num_backwards, is_VAE, VAE_beta, is_uncertainty_net, lr, reg_amp, activation_gen, activation_model, get_struct_str(struct_param_gen_base), optim_mode, loss_core, exp_id)
make_dir(filename)
print(filename)



# Obtain nets:
net = Net(input_size = input_size,
          struct_param = [[num_neurons, "Simple_Layer", {}] for num_neurons in list(main_hidden_neurons) + [output_size]],
          settings = {"activation": activation_model},
         )

# Setting up optimizer and loss functions:
optimizer = optim.Adam(net.parameters(), lr = lr)

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
        criterion = loss_fun_core
early_stopping = Early_Stopping(patience = patience)

# Obtain tasks:
if "tasks_train" not in locals():
    if is_load_data:
        try:
            dataset = pickle.load(open(filename + "data.p", "rb"))
            tasks_train = dataset["tasks_train"]
            tasks_test = dataset["tasks_test"]
            print("dataset loaded.")
        except:
            print("dataset do not exist. Create one")
            tasks_train, tasks_test = get_tasks(task_id_list, num_train_tasks, num_test_tasks, task_settings = task_settings, is_cuda = is_cuda)
            dataset = {"tasks_train": tasks_train, "tasks_test": tasks_test}
            pickle.dump(dataset, open(filename + "data.p", "wb"))
    else:
        tasks_train, tasks_test = get_tasks(task_id_list, num_train_tasks, num_test_tasks, task_settings = task_settings, is_cuda = is_cuda)
        dataset = {"tasks_train": tasks_train, "tasks_test": tasks_test}
        pickle.dump(dataset, open(filename + "data.p", "wb"))
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

# Training:
for i in range(num_iter + 1):
    chosen_task_keys = np.random.choice(list(tasks_train.keys()), batch_size_task, replace = False).tolist()
    if optim_mode == "indi":
        for task_key, task in tasks_train.items():
            if task_key not in chosen_task_keys:
                continue
            ((X_train, y_train), (X_test, y_test)), _ = task
            for k in range(num_backwards):
                optimizer.zero_grad()
                y_pred = net(X_train)
                loss = criterion(y_pred, y_train)
                reg = get_reg(reg_dict, net = net, is_cuda = is_cuda)
                loss = loss + reg
                loss.backward(retain_graph = True)
                optimizer.step()
    elif optim_mode == "sum":
        optimizer.zero_grad()
        loss_total = Variable(torch.FloatTensor([0]), requires_grad = False)
        if is_cuda:
            loss_total = loss_total.cuda()
        for task_key, task in tasks_train.items():
            if task_key not in chosen_task_keys:
                continue
            ((X_train, y_train), (X_test, y_test)), _ = task
            y_pred = net(X_train)
            loss = criterion(y_pred, y_train)
            reg = get_reg(reg_dict, net = net, is_cuda = is_cuda)
            loss_total = loss_total + loss + reg
        loss_total.backward()
        optimizer.step()
    else:
        raise Exception("optim_mode {0} not recognized!".format(optim_mode))
    

    loss_test_record = []
    for task_key, task in tasks_test.items():
        (_, (X_test, y_test)), _ = task
        y_pred_test = net(X_test)
        loss_test = criterion(y_pred_test, y_test).data[0]
        loss_test_record.append(loss_test)
    to_stop = early_stopping.monitor(np.mean(loss_test_record))

    # Validation and visualization:
    if i % inspect_interval == 0 or to_stop:
        print("=" * 50)
        print("training tasks:")
        for task_key, task in tasks_train.items():
            (_, (X_test, y_test)), _ = task
            y_pred_test = net(X_test)
            loss_test = loss_test_sampled = criterion(y_pred_test, y_test).data[0]
            mse = nn.MSELoss()(y_pred_test, y_test).data[0]
            reg = get_reg(reg_dict, net = net, is_cuda = is_cuda).data[0]
            KLD_test = 0
            data_record["loss"][task_key].append(loss_test)
            data_record["loss_sampled"][task_key].append(loss_test_sampled)
            data_record["mse"][task_key].append(mse)
            data_record["reg"][task_key].append(reg)
            data_record["KLD"][task_key].append(KLD_test)
            print('{0}\ttrain\t{1}  \tloss: {2:.5f}\tloss_sampled:{3:.5f} \tmse:{4:.5f}\tKLD:{5:.6f}\treg:{6:.6f}'.format(i, task_key, loss_test, loss_test_sampled, mse, KLD_test, reg))
        for task_key, task in tasks_test.items():
            (_, (X_test, y_test)), _ = task
            y_pred_test = net(X_test)
            loss_test = loss_test_sampled = criterion(y_pred_test, y_test).data[0]
            mse = nn.MSELoss()(y_pred_test, y_test).data[0]
            reg = get_reg(reg_dict, net = net, is_cuda = is_cuda).data[0]
            KLD_test = 0
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
#         mse_few_shot = plot_few_shot_loss(master_model, tasks_test, isplot = isplot)
        record_data(data_record, 
                    [np.mean(loss_train_list), np.median(loss_train_list), np.mean(reg_train_list), i,
                     np.mean(loss_test_list), np.median(loss_test_list), np.mean(reg_test_list),
                     np.mean(loss_train_sampled_list), np.median(loss_train_sampled_list), 
                     np.mean(loss_test_sampled_list), np.median(loss_test_sampled_list),
                     np.mean(mse_train_list), np.median(mse_train_list), 
                     np.mean(mse_test_list), np.median(mse_test_list), 
#                      mse_few_shot,
                    ], 
                    ["loss_mean_train", "loss_median_train", "reg_mean_train", "iter",
                     "loss_mean_test", "loss_median_test", "reg_mean_test",
                     "loss_sampled_mean_train", "loss_sampled_median_train",
                     "loss_sampled_mean_test", "loss_sampled_median_test", 
                     "mse_mean_train", "mse_median_train", "mse_mean_test", "mse_median_test", 
#                      "mse_few_shot",
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
#         statistics_list_train, z_list_train = plot_task_ensembles(tasks_train, statistics_Net, generative_Net, is_VAE = is_VAE, is_regulated_net = is_regulated_net, title = "y_pred_train vs. y_train", isplot = isplot)
#         statistics_list_test, z_list_test = plot_task_ensembles(tasks_test, statistics_Net, generative_Net, is_VAE = is_VAE, is_regulated_net = is_regulated_net, title = "y_pred_test vs. y_test", isplot = isplot)
#         record_data(data_record, [np.array(z_list_train), np.array(z_list_test), np.array(statistics_list_train), np.array(statistics_list_test)], 
#                     ["z_list_train_list", "z_list_test_list", "statistics_list_train_list", "statistics_list_test_list"])
        if isplot:
#             print("train statistics vs. z:")
#             plot_statistics_vs_z(z_list_train, statistics_list_train)
#             print("test statistics vs. z:")
#             plot_statistics_vs_z(z_list_test, statistics_list_test)

            # Plotting individual test data:
            if "bounce" in task_id_list[0]:
                plot_individual_tasks_bounce(tasks_test, num_examples_show = 40, num_tasks_show = 6, model = net, num_shots = 200)
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
        record_data(info_dict, [net.model_dict, i], ["model_dict", "iter"])
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
