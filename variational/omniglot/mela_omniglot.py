
# coding: utf-8

# In[1]:


import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD, Adam

from task import OmniglotTask, MNISTTask
from dataset import Omniglot, MNIST
from data_loading import get_data_loader
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..', ".."))
from mela.variational.variational_meta_learning import Statistics_Net_Conv, Generative_Net_Conv, Master_Model, load_model_dict
from mela.pytorch.util_pytorch import get_num_params, to_Variable, to_np_array
from mela.util import plot_matrices, make_dir

is_cuda = torch.cuda.is_available()

def get_task(root, n_cl, n_inst, split='train'):
    if 'mnist' in root:
        return MNISTTask(root, n_cl, n_inst, split)
    elif 'omniglot' in root:
        return OmniglotTask(root, n_cl, n_inst, split)
    else:
        print('Unknown dataset')
        raise(Exception)

def get_metrics(master_model, X_test, y_test, loss_fn):
    y_logit = master_model(X_test)
    y_pred = y_logit.max(1)[1]
    acc = (to_np_array((y_test == y_pred).float().sum()) / len(y_test))[0]
    loss = to_np_array(loss_fn(y_logit, y_test))[0]
    return loss, acc


# ## Prepare MeLA:

# In[3]:


pre_pooling_neurons = 200
statistics_output_neurons = 20
num_classes = 5
input_channels = 3
activation_default = "leakyRelu"
activation_conv = "leakyReluFlat"

struct_param_pre_conv = [
    [8, "Conv2d", {"kernel_size": 4, "stride": 2, "activation": activation_conv}],
#     [None, "MaxPool2d", {"kernel_size": 2, "return_indices": False}],
    [4, "Conv2d", {"kernel_size": 3, "stride": 1, "activation": activation_conv}],
    [4, "Conv2d", {"kernel_size": 3, "stride": 1, "activation": activation_conv}],
    [40, "Simple_Layer", {"activation": "linear", "layer_input_size": 324}]
]
struct_param_pre = [[60, "Simple_Layer", {"activation": activation_default}],
                    [pre_pooling_neurons, "Simple_Layer", {"activation": "linear"}],
                   ]
struct_param_post = [[60, "Simple_Layer", {"activation": activation_default}],
                     [60, "Simple_Layer", {"activation": activation_default}],
                     [statistics_output_neurons, "Simple_Layer", {"activation": "linear"}],
                    ]
struct_param_gen_base = [[40, "Simple_Layer", {"activation": activation_default}],
                         [10, "Simple_Layer", {"activation": "linear"}],
                        ]
struct_param_model = [[64, "Conv2d", {"kernel_size": 3, "stride": 1, "dilation": 2}], 
                      [64, "BatchNorm2d", {"activation": activation_conv}],
                     ] * 4 + \
                     [[num_classes, "Simple_Layer", {"activation": "linear"}]]
main_weight_neurons = [3*64*3*3, 64, 64*64*3*3, 64, 64*64*3*3, 64, 64*64*3*3, 64, 9216 * 5]
main_bias_neurons = [64, 64, 64, 64, 64, 64, 64, 64, 5]

W_struct_param_list = []
b_struct_param_list = []
for i, num_weight_neurons in enumerate(main_weight_neurons):
    struct_param_weight = struct_param_gen_base + [[num_weight_neurons, "Simple_Layer", {"activation": "linear"}]]
    struct_param_bias = struct_param_gen_base + [[main_bias_neurons[i], "Simple_Layer", {"activation": "linear"}]]
    W_struct_param_list.append(struct_param_weight)
    b_struct_param_list.append(struct_param_bias)

statistics_Net_Conv = Statistics_Net_Conv(input_channels = input_channels,
                                          num_classes = num_classes,
                                          pre_pooling_neurons = pre_pooling_neurons,
                                          struct_param_pre_conv = struct_param_pre_conv, 
                                          struct_param_pre = struct_param_pre,
                                          struct_param_post = struct_param_post,
                                          is_cuda = is_cuda,
                                         )
generative_Net_Conv = Generative_Net_Conv(input_channels = input_channels,
                                          latent_size = statistics_output_neurons,
                                          W_struct_param_list = W_struct_param_list,
                                          b_struct_param_list = b_struct_param_list,
                                          struct_param_model = struct_param_model,
                                          is_cuda = is_cuda,
                                         )
master_model = Master_Model(statistics_Net = statistics_Net_Conv, generative_Net = generative_Net_Conv, is_cuda = is_cuda)
print("Num_params: {0}".format(get_num_params(master_model)))


# ## Train:

# In[ ]:


optim_mode = "indi"
dataset='omniglot'
num_inst=6
meta_batch_size=20
num_updates=15000
lr=1e-1
meta_lr=1e-3
loss_fn = nn.CrossEntropyLoss() 
reg_amp = 1e-9
exp='maml-omniglot-{0}way-{1}shot-TEST'.format(num_classes, num_inst)
make_dir("output/{0}/".format(exp))

random.seed(1337)
np.random.seed(1337)

tr_loss, tr_acc, val_loss, val_acc = [], [], [], []
mtr_loss, mtr_acc, mval_loss, mval_acc = [], [], [], []
reg_list = []

optimizer = torch.optim.Adam(master_model.parameters(), lr = meta_lr)
for i in range(num_updates):
    # Evaluate on test tasks
#     mt_loss, mt_acc, mv_loss, mv_acc = test()
#     mtr_loss.append(mt_loss)
#     mtr_acc.append(mt_acc)
#     mval_loss.append(mv_loss)
#     mval_acc.append(mv_acc)
    # Collect a meta batch update
    grads = []
    tloss, tacc, vloss, vacc, reg_batch = 0.0, 0.0, 0.0, 0.0, 0.0
    if optim_mode == "indi":
        for k in range(meta_batch_size):
            # Get data:
            task = get_task('../data/{}'.format(dataset), num_classes, num_inst)
            train_loader = get_data_loader(task, batch_size = num_inst, split='train')
            val_loader = get_data_loader(task, batch_size = num_inst, split='val')
            X_train, y_train = train_loader.__iter__().next()
            X_test, y_test = val_loader.__iter__().next()
            X_train, y_train, X_test, y_test = to_Variable(X_train, y_train, X_test, y_test, is_cuda = is_cuda)
            
            # Get gradient:
            optimizer.zero_grad()
            results = master_model.get_predictions(X_test, X_train, y_train, is_time_series = False)
            loss = loss_fn(results["y_pred"], y_test)
            reg = master_model.get_regularization(source = ["weight", "bias", "W_gen", "b_gen"], target = ["statistics_Net", "generative_Net"]) * reg_amp
            loss = loss + reg
            loss.backward()
            
            # Get metrics:
            master_model.get_statistics(X_train, y_train)
            trl, tra = get_metrics(master_model, X_train, y_train, loss_fn)
            vall, vala = get_metrics(master_model, X_test, y_test, loss_fn)
            tloss += trl
            tacc += tra
            vloss += vall
            vacc += vala
            reg_batch += to_np_array(reg)[0]
            
            # Gradient descent:
            optimizer.step()
            

    elif optim_mode == "sum":
        optimizer.zero_grad()
        loss_total = Variable(torch.FloatTensor([0]), requires_grad = False)
        if is_cuda:
            loss_total = loss_total.cuda()
        
        for i in range(meta_batch_size):
            # Get data:
            task = get_task('../data/{}'.format(dataset), num_classes, num_inst)
            train_loader = get_data_loader(task, batch_size = num_inst, split='train')
            val_loader = get_data_loader(task, batch_size = num_inst, split='val')
            X_train, y_train = train_loader.__iter__().next()
            X_test, y_test = val_loader.__iter__().next()
            X_train, y_train, X_test, y_test = to_Variable(X_train, y_train, X_test, y_test, is_cuda = is_cuda)
            
            # Get single-task loss:
            optimizer.zero_grad()
            results = master_model.get_predictions(X_test, X_train, y_train, is_time_series = False)
            loss = loss_fn(results["y_pred"], y_test)
            reg = master_model.get_regularization(source = ["weight", "bias", "W_gen", "b_gen"], target = ["statistics_Net", "generative_Net"]) * reg_amp
            loss_total = loss_total + loss + reg

            # Get metrics:
            master_model.generative_Net.set_latent_param(results["statistics"])
            trl, tra = get_metrics(master_model, X_train, y_train, loss_fn)
            vall, vala = get_metrics(master_model, X_test, y_test, loss_fn)
            tloss += trl
            tacc += tra
            vloss += vall
            vacc += vala
            reg_batch += to_np_array(reg)[0]
        
        # Gradient descient on the sum of loss:
        loss_total.backward()
        optimizer.step()
    else:
        raise Exception("optim_mode {0} not recognized!".format(optim_mode))
    
    # Save stuff
    tr_loss.append(tloss / meta_batch_size)
    tr_acc.append(tacc / meta_batch_size)
    val_loss.append(vloss / meta_batch_size)
    val_acc.append(vacc / meta_batch_size)
    reg_list.append(reg_batch / meta_batch_size)
    
    print("iter {0}\ttrain_loss: {1:.4f}\ttest_loss: {2:.4f}\ttrain_acc: {3:.4f}\ttest_acc: {4:.4f}\treg: {5:.6f}".format(i, tr_loss[-1], val_loss[-1], tr_acc[-1], val_acc[-1], reg_list[-1]))

    np.save('output/{}/tr_loss.npy'.format(exp), np.array(tr_loss))
    np.save('output/{}/tr_acc.npy'.format(exp), np.array(tr_acc))
    np.save('output/{}/val_loss.npy'.format(exp), np.array(val_loss))
    np.save('output/{}/val_acc.npy'.format(exp), np.array(val_acc))

#     np.save('output/{}/meta_tr_loss.npy'.format(exp), np.array(mtr_loss))
#     np.save('output/{}/meta_tr_acc.npy'.format(exp), np.array(mtr_acc))
#     np.save('output/{}/meta_val_loss.npy'.format(exp), np.array(mval_loss))
#     np.save('output/{}/meta_val_acc.npy'.format(exp), np.array(mval_acc))

