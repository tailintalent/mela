
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
from AI_scientist.settings.filepath import dataset_PATH
from AI_scientist.pytorch.net import Net
from AI_scientist.pytorch.util_pytorch import Loss_with_uncertainty
from AI_scientist.variational.variational_meta_learning import Master_Model, Statistics_Net, Generative_Net, load_model_dict, get_regulated_statistics
from AI_scientist.variational.variational_meta_learning import VAE_Loss, sample_Gaussian, clone_net, get_nets, get_tasks, evaluate, get_reg, load_trained_models
from AI_scientist.variational.variational_meta_learning import plot_task_ensembles, plot_individual_tasks, plot_statistics_vs_z, plot_data_record, get_corrcoef
from AI_scientist.variational.variational_meta_learning import plot_few_shot_loss, plot_individual_tasks_bounce
from AI_scientist.variational.variational_meta_learning import get_latent_model_data, get_polynomial_class, get_Legendre_class, get_master_function


# ## Task generation:

# In[5]:


task_id = "C-tanh"
task_id = "C-sin"

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
def get_numpy_tasks(tasks):
    tasks_save = []
    for task_key, task in tasks.items():
        ((X_train, y_train), (X_test, y_test)), z_info = task
        tasks_save.append([[[X_train.data.numpy(), y_train.data.numpy()], [X_test.data.numpy(), y_test.data.numpy()]], z_info])
    return tasks_save

if task_id == "C-sin":
    task_id_list = ["C-sin"]
    task_settings = {"test_size": 0.5, "num_examples": 20}
    num_train_tasks = 50
    num_test_tasks = 5000
elif task_id == "C-tanh":
    task_id_list = ["C-tanh"]
    task_settings = {"test_size": 0.5, "num_examples": 20}
    num_train_tasks = 50
    num_test_tasks = 5000
else:
    raise
tasks_train, tasks_test = get_tasks(task_id_list, num_train_tasks, num_test_tasks, task_settings = task_settings)
filename = dataset_PATH + task_id + ".p"

tasks = {"tasks_train": get_numpy_tasks(tasks_train),
         "tasks_test": get_numpy_tasks(tasks_test),
         }
pickle.dump(tasks, open(filename, "wb"))


# ## Load_tasks:

# In[2]:


task_id = "C-tanh"
# task_id = "C-sin"

filename = dataset_PATH + task_id + ".p"
tasks = pickle.load(open(filename, "rb"))

tasks_train = tasks["tasks_train"]
tasks_test = tasks["tasks_test"]


# In[3]:


len(tasks_train)


# In[4]:


len(tasks_test)


# In[ ]:


# Train with training tasks:
for task in tasks_train:
    ((X_train, y_train), (X_test, y_test)), _ = task
   

# Validate with testing tasks:
# This is only for tanh/sin tasks, where we partition the 5000 testing tasks into 100 x evaluations, and each evaluation has 50 testing tasks:
loss_list = []
for i in range(100):
    tasks_test_iter = tasks_test[i * 50 : (i + 1) * 50]
    # Perform evaluation, accumulate the loss:
    loss = evaluate(tasks_test_iter)
    loss_list.append(loss)

# Then obtain mean and std:

