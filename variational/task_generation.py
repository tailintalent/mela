
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
from AI_scientist.settings.filepath import dataset_PATH_short
from AI_scientist.pytorch.net import Net
from AI_scientist.pytorch.util_pytorch import Loss_with_uncertainty
from AI_scientist.variational.util_variational import get_numpy_tasks, get_torch_tasks
from AI_scientist.variational.variational_meta_learning import Master_Model, Statistics_Net, Generative_Net, load_model_dict, get_regulated_statistics
from AI_scientist.variational.variational_meta_learning import VAE_Loss, sample_Gaussian, clone_net, get_nets, get_tasks, evaluate, get_reg, load_trained_models
from AI_scientist.variational.variational_meta_learning import plot_task_ensembles, plot_individual_tasks, plot_statistics_vs_z, plot_data_record, get_corrcoef
from AI_scientist.variational.variational_meta_learning import plot_few_shot_loss, plot_individual_tasks_bounce
from AI_scientist.variational.variational_meta_learning import get_latent_model_data, get_polynomial_class, get_Legendre_class, get_master_function


# ## Task generation:

# In[ ]:


# task_id = "C-tanh"
# task_id = "C-sin"
task_id = "bounce-states"
# task_id = "bounce-images"
seed = 1

np.random.seed(seed)
torch.manual_seed(seed)

if task_id in ["C-sin", "C-tanh"]:
    num_shots = 10
    num_train_tasks = 100
    num_test_tasks = 20000
elif task_id == "bounce-states":
    num_shots = 100
    num_train_tasks = 100
    num_test_tasks = 1000
elif task_id == "bounce-images":
    num_shots = 100
    num_train_tasks = 100
    num_test_tasks = 100
else:
    raise
task_settings = {"test_size": 0.5, "num_examples": num_shots * 2}
tasks_train, tasks_test = get_tasks([task_id], num_train_tasks, num_test_tasks, task_settings = task_settings, forward_steps = list(range(1,11)))
filename = dataset_PATH_short + task_id + "_{0}-shot.p".format(num_shots)

tasks = {"tasks_train": get_numpy_tasks(tasks_train),
         "tasks_test": get_numpy_tasks(tasks_test),
         }
pickle.dump(tasks, open(filename, "wb"))


# ## Load_tasks:

# In[8]:


# task_id = "C-tanh"
# task_id = "C-sin"
# task_id = "bounce-states"
task_id = "bounce-images"
if task_id in ["C-sin", "C-tanh"]:
    num_shots = 10
elif task_id in ["bounce-states", "bounce-images"]:
    num_shots = 100

filename = dataset_PATH_short + task_id + "_{0}-shot.p".format(num_shots)
tasks = pickle.load(open(filename, "rb"))

tasks_train = tasks["tasks_train"]
tasks_test = tasks["tasks_test"]


# In[ ]:


# Train with training tasks:
for task in tasks_train:
    ((X_train, y_train), (X_test, y_test)), _ = task

# Validate with testing tasks:
# This is only for tanh/sin tasks, where we partition the 20000 testing tasks into 200 x evaluations, and each evaluation has 100 testing tasks:
loss_list = []
for i in range(100):
    tasks_test_iter = tasks_test[i * 100 : (i + 1) * 100]
    # Perform evaluation, accumulate the loss:
    loss = evaluate(tasks_test_iter)
    loss_list.append(loss)

# Then obtain mean and std:

