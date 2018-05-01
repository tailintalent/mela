import numpy as np
import pickle
import numpy as np
import pickle
from itertools import chain
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.max_columns = 50
from IPython.display import display, HTML
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from AI_scientist.prepare_dataset import Dataset_Gen
from AI_scientist.util import plot_matrices, make_dir, get_struct_str, get_args, Early_Stopping, record_data, manifold_embedding, get_args, make_dir, sort_two_lists
from AI_scientist.settings.filepath import variational_model_PATH, dataset_PATH
from AI_scientist.pytorch.model import Model, load_model
from AI_scientist.pytorch.net import Net
from AI_scientist.variational.variational_meta_learning import Master_Model, Statistics_Net, Generative_Net, load_model_dict
from AI_scientist.variational.variational_meta_learning import VAE_Loss, sample_Gaussian, clone_net, get_nets, get_tasks, evaluate, get_reg, load_trained_models, get_relevance
from AI_scientist.variational.variational_meta_learning import plot_task_ensembles, plot_individual_tasks, plot_statistics_vs_z, plot_data_record
from AI_scientist.variational.variational_meta_learning import get_latent_model_data, get_polynomial_class, get_Legendre_class, get_master_function
import torch
import torch.nn as nn
from torch.autograd import Variable



if __name__ == "__main__":
    exp_id = "exp1.1" # Standard
    filename = "Net_['master_tanh']_input_1_(100,100)_statistics-neurons_8_pre_100_pooling_max_context_0_batch_100_backward_1_VAE_False_1.0_lr_1e-05_reg_0.0_actgen_elu_actmodel_leakyRelu_struct_60Si-60Si-60Si_exp1.1_" # Best
    filename = "Net_['master_tanh']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_0_batch_100_backward_1_VAE_False_1.0_lr_0.0001_reg_1e-06_actgen_elu_actmodel_leakyRelu_struct_60Si-60Si-60Si_exp1.1_" # second best
    # filename = "Net_['master_tanh']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_0_batch_100_backward_1_VAE_False_1.0_lr_0.0001_reg_0.0_actgen_elu_actmodel_leakyRelu_struct_60Si-60Si-60Si_exp1.1_"
    filename = "Net_['master_tanh']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_0_batch_100_backward_1_VAE_False_1.0_lr_1e-05_reg_1e-07_actgen_elu_actmodel_leakyRelu_struct_60Si-60Si-60Si_exp1.1_"
    
    exp_id = "exp1.2" # Standard
    filename = "Net_['master_tanh']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_0_batch_100_backward_1_VAE_True_0.2_lr_0.0001_reg_1e-06_actgen_elu_actmodel_leakyRelu_struct_60Si-60Si-60Si_exp1.2_"
    # filename = "Net_['master_tanh']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_0_batch_100_backward_1_VAE_True_0.2_lr_1e-05_reg_0.0_actgen_elu_actmodel_leakyRelu_struct_60Si-60Si-60Si_exp1.2_"
    # filename = "Net_['master_tanh']_input_1_(100,100)_statistics-neurons_4_pre_50_pooling_max_context_0_batch_100_backward_1_VAE_True_0.2_lr_0.0001_reg_0.0_actgen_elu_actmodel_leakyRelu_struct_60Si-60Si-60Si_exp1.2_"
    # filename = "Net_['master_tanh']_input_1_(100,100)_statistics-neurons_4_pre_50_pooling_max_context_0_batch_100_backward_1_VAE_True_0.2_lr_1e-05_reg_1e-06_actgen_elu_actmodel_leakyRelu_struct_60Si-60Si-60Si_exp1.2_"
    # filename = "Net_['master_tanh']_input_1_(100,100)_statistics-neurons_4_pre_50_pooling_max_context_0_batch_100_backward_1_VAE_True_0.2_lr_2e-06_reg_1e-07_actgen_elu_actmodel_leakyRelu_struct_60Si-60Si-60Si_exp1.2_"
    # filename = "Net_['master_tanh']_input_1_(100,100)_statistics-neurons_4_pre_50_pooling_max_context_0_batch_100_backward_1_VAE_True_0.2_lr_2e-06_reg_0.0_actgen_elu_actmodel_leakyRelu_struct_60Si-60Si-60Si_exp1.2_"
    # filename = "Net_['master_tanh']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_0_batch_100_backward_1_VAE_True_0.2_lr_1e-05_reg_1e-07_actgen_elu_actmodel_leakyRelu_struct_60Si-60Si-60Si_exp1.2_"
    # filename = "Net_['master_tanh']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_0_batch_50_backward_1_VAE_True_0.2_lr_0.0001_reg_1e-07_actgen_elu_actmodel_leakyRelu_struct_60Si-60Si-60Si_exp1.2_"
    # filename = "Net_['master_tanh']_input_1_(100,100)_statistics-neurons_4_pre_50_pooling_max_context_0_batch_100_backward_1_VAE_True_0.2_lr_2e-06_reg_1e-06_actgen_elu_actmodel_leakyRelu_struct_60Si-60Si-60Si_exp1.2_"
    # filename = "Net_['master_tanh']_input_1_(100,100)_statistics-neurons_4_pre_50_pooling_max_context_0_batch_50_backward_1_VAE_True_0.2_lr_1e-05_reg_0.0_actgen_elu_actmodel_leakyRelu_struct_60Si-60Si-60Si_exp1.2_"

    # exp_id = "exp1.3"  # Gaussian standard
    # filename = "Net_['master_Gaussian']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_0_batch_100_backward_1_VAE_False_1.0_lr_0.0001_reg_0.0_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp1.3_"
    # filename = "Net_['master_Gaussian']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_3_batch_50_backward_1_VAE_False_1.0_lr_0.001_reg_0.0_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_sum_exp1.3_"
    # filename = "Net_['master_Gaussian']_input_1_(100,100)_statistics-neurons_4_pre_50_pooling_max_context_3_batch_50_backward_1_VAE_False_1.0_lr_0.0001_reg_0.0_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp1.3_"
    # filename = "Net_['master_Gaussian']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_0_batch_100_backward_1_VAE_False_1.0_lr_0.001_reg_1e-06_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_sum_exp1.3_"
    # filename = "Net_['master_Gaussian']_input_1_(100,100)_statistics-neurons_4_pre_50_pooling_max_context_0_batch_50_backward_1_VAE_False_1.0_lr_0.0001_reg_1e-06_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp1.3_"
    # filename = "Net_['master_Gaussian']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_3_batch_100_backward_1_VAE_False_1.0_lr_0.0001_reg_1e-07_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp1.3_"
    # filename = "Net_['master_Gaussian']_input_1_(100,100)_statistics-neurons_4_pre_50_pooling_max_context_3_batch_100_backward_1_VAE_False_1.0_lr_0.0001_reg_0.0_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp1.3_"
    # filename = "Net_['master_Gaussian']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_3_batch_100_backward_1_VAE_False_1.0_lr_0.0001_reg_0.0_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp1.3_"
    # filename = "Net_['master_Gaussian']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_3_batch_100_backward_1_VAE_False_1.0_lr_0.001_reg_1e-06_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp1.3_"
    # filename = "Net_['master_Gaussian']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_0_batch_100_backward_1_VAE_False_1.0_lr_0.001_reg_0.0_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp1.3_"
    
    # exp_id = "exp1.32"  # Gaussian VAE
    # filename = "Net_['master_Gaussian']_input_1_(100,100)_statistics-neurons_4_pre_50_pooling_mean_context_0_batch_50_backward_1_VAE_True_0.2_lr_0.0001_reg_0.0_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp1.32_"
    # filename = "Net_['master_Gaussian']_input_1_(100,100)_statistics-neurons_4_pre_50_pooling_max_context_0_batch_50_backward_1_VAE_True_0.05_lr_0.0001_reg_0.0_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp1.32_"
    # filename = "Net_['master_Gaussian']_input_1_(100,100)_statistics-neurons_4_pre_50_pooling_max_context_0_batch_100_backward_1_VAE_True_0.05_lr_0.001_reg_1e-07_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp1.32_"
    # filename = "Net_['master_Gaussian']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_0_batch_100_backward_1_VAE_True_0.05_lr_0.0001_reg_1e-06_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp1.32_"
    # filename = "Net_['master_Gaussian']_input_1_(100,100)_statistics-neurons_4_pre_50_pooling_max_context_3_batch_100_backward_1_VAE_True_0.05_lr_0.0001_reg_1e-06_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp1.32_"
    # filename = "Net_['master_Gaussian']_input_1_(100,100)_statistics-neurons_4_pre_50_pooling_mean_context_3_batch_100_backward_1_VAE_True_0.05_lr_1e-05_reg_1e-07_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp1.32_"
    # filename = "Net_['master_Gaussian']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_3_batch_50_backward_1_VAE_True_0.05_lr_0.001_reg_1e-06_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp1.32_"
    # filename = "Net_['master_Gaussian']_input_1_(100,100)_statistics-neurons_4_pre_50_pooling_mean_context_3_batch_50_backward_1_VAE_True_0.05_lr_0.0001_reg_1e-06_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_sum_exp1.32_"
    # filename = "Net_['master_Gaussian']_input_1_(100,100)_statistics-neurons_4_pre_50_pooling_max_context_0_batch_100_backward_1_VAE_True_0.05_lr_0.001_reg_0.0_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp1.32_"
    # filename = "Net_['master_Gaussian']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_0_batch_50_backward_1_VAE_True_0.05_lr_0.0001_reg_1e-07_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp1.32_"

    # exp_id = "exp1.4" # Standard, testing optim_type
    # filename = "Net_['master_tanh']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_0_batch_100_backward_1_VAE_False_1.0_lr_0.001_reg_1e-06_actgen_elu_actmodel_leakyRelu_struct_60Si-60Si-60Si_sum_exp1.4"
    # filename = "Net_['master_tanh']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_0_batch_100_backward_1_VAE_False_1.0_lr_0.0001_reg_1e-06_actgen_elu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp1.4"
    # filename = "Net_['master_tanh']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_0_batch_50_backward_1_VAE_False_1.0_lr_0.0001_reg_1e-06_actgen_elu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp1.4"
    # filename = "Net_['master_tanh']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_3_batch_50_backward_1_VAE_False_1.0_lr_0.001_reg_0.0_actgen_elu_actmodel_leakyRelu_struct_60Si-60Si-60Si_sum_exp1.4"
    # filename = "Net_['master_tanh']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_3_batch_50_backward_1_VAE_False_1.0_lr_0.0001_reg_0.0_actgen_elu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp1.4"
    # filename = "Net_['master_tanh']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_3_batch_100_backward_1_VAE_False_1.0_lr_0.001_reg_1e-07_actgen_elu_actmodel_leakyRelu_struct_60Si-60Si-60Si_sum_exp1.4"
    # filename = "Net_['master_tanh']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_0_batch_50_backward_1_VAE_False_1.0_lr_0.001_reg_1e-06_actgen_elu_actmodel_leakyRelu_struct_60Si-60Si-60Si_sum_exp1.4"
    # filename = "Net_['master_tanh']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_0_batch_50_backward_1_VAE_False_1.0_lr_0.0001_reg_1e-07_actgen_elu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp1.4"
    # filename = "Net_['master_tanh']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_0_batch_50_backward_1_VAE_False_1.0_lr_0.001_reg_1e-07_actgen_elu_actmodel_leakyRelu_struct_60Si-60Si-60Si_sum_exp1.4"
    # filename = "Net_['master_tanh']_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_3_batch_50_backward_1_VAE_False_1.0_lr_0.0001_reg_1e-06_actgen_elu_actmodel_leakyRelu_struct_60Si-60Si-60Si_sum_exp1.4"
    
    # exp_id = "exp2.0"
    # filename_list = [
    #  "Net_('master_tanh',)_input_1_(100,100)_statistics-neurons_4_pre_50_pooling_max_context_3_batch_100_backward_1_VAE_False_1.0_uncertainty_True_lr_0.0001_reg_0.0_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp2.0_",
    #  "Net_('master_tanh',)_input_1_(100,100)_statistics-neurons_4_pre_50_pooling_max_context_0_batch_100_backward_1_VAE_False_1.0_uncertainty_True_lr_0.0001_reg_0.0_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp2.0_",
    #  "Net_('master_tanh',)_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_0_batch_100_backward_1_VAE_False_1.0_uncertainty_True_lr_0.0001_reg_1e-07_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp2.0_",
    #  "Net_('master_tanh',)_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_0_batch_100_backward_1_VAE_False_1.0_uncertainty_True_lr_0.0001_reg_0.0_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp2.0_",
    #  "Net_('master_tanh',)_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_3_batch_100_backward_1_VAE_False_1.0_uncertainty_True_lr_0.0001_reg_1e-07_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp2.0_",
    #  "Net_('master_tanh',)_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_0_batch_100_backward_1_VAE_False_1.0_uncertainty_True_lr_0.0005_reg_0.0_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp2.0_",
    #  "Net_('master_tanh',)_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_3_batch_100_backward_1_VAE_False_1.0_uncertainty_True_lr_0.0005_reg_1e-07_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp2.0_",
    #  "Net_('master_tanh',)_input_1_(100,100)_statistics-neurons_4_pre_50_pooling_max_context_0_batch_100_backward_1_VAE_False_1.0_uncertainty_True_lr_0.0005_reg_1e-07_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp2.0_",
    #  "Net_('master_tanh',)_input_1_(100,100)_statistics-neurons_4_pre_100_pooling_max_context_0_batch_100_backward_1_VAE_False_1.0_uncertainty_True_lr_0.0005_reg_1e-07_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp2.0_",
    #  "Net_('master_tanh',)_input_1_(100,100)_statistics-neurons_4_pre_50_pooling_max_context_3_batch_100_backward_1_VAE_False_1.0_uncertainty_True_lr_0.0005_reg_0.0_actgen_leakyRelu_actmodel_leakyRelu_struct_60Si-60Si-60Si_individual_exp2.0_",
    # ]
    # filename = filename_list[0]
    is_model_dict = True

    print(filename)
    filename = variational_model_PATH + "/trained_models/{0}_good/".format(exp_id) + filename
    if is_model_dict:
        info_dict = pickle.load(open(filename + ".p", "rb"))
        master_model = load_model_dict(info_dict["model_dict"][-1])
        statistics_Net = master_model.statistics_Net
        generative_Net = master_model.generative_Net
        data_record = info_dict["data_record"]
    else:
        statistics_Net, generative_Net, data_record = load_trained_models(filename)

    filename_split = filename.split("_")
    task_id_list = eval("_".join(filename_split[filename_split.index('good/Net') + 1: filename_split.index("input")]))
    task_settings = data_record["task_settings"][0]
    print("task_settings:\n", task_settings)
    is_VAE = eval(filename_split[filename_split.index("VAE") + 1])
    is_uncertainty_net = eval(filename_split[filename_split.index("uncertainty") + 1]) if "uncertainty" in filename_split else False
    num_test_tasks = 1000
    task_settings["num_examples"] = 2000
    task_settings["test_size"] = 0.95
    tasks_train, tasks_test = get_tasks(task_id_list, 1, num_test_tasks, task_settings = task_settings)
    plot_types = ["standard"]
    # plot_types = ["gradient"]
    plot_types = ["slider"]

    for plot_type in plot_types:
        if plot_type == "standard":
            plot_data_record(data_record, is_VAE = is_VAE)

            statistics_list_test, z_list_test = plot_task_ensembles(tasks_test, statistics_Net, generative_Net, is_VAE = is_VAE, title = "y_pred_test vs. y_test")
            print("test statistics vs. z:")
            plot_statistics_vs_z(z_list_test, statistics_list_test)
            _ = plot_individual_tasks(tasks_test, statistics_Net, generative_Net, is_VAE = is_VAE, xlim = task_settings["xlim"])

        elif plot_type == "gradient":
            batch_size = 256
            sample_task_id = task_id_list[0] + "_{0}".format(np.random.randint(num_test_tasks))
            print("sample_task_id: {0}".format(sample_task_id))
            ((X_train, y_train), (X_test, y_test)), _ = tasks_test[sample_task_id]
            epochs_statistics = 50
            lr_statistics = 5e-3
            optim_type_statistics = "LBFGS"

            # epochs = 50
            # lr = 1e-3
            # optimizer = "adam"

            epochs = 50
            lr = 5e-3
            optim_type = "LBFGS"

            master_model.get_statistics(X_train, y_train)
            y_pred = master_model(X_test)
            loss_list_0 = master_model.latent_param_quick_learn(X_train, y_train, validation_data = (X_test, y_test), batch_size = batch_size, 
                                                                epochs = epochs_statistics, lr = lr_statistics, optim_type = optim_type_statistics)
            y_pred_new = master_model(X_test)
            plt.plot(X_test.data.numpy(), y_test.data.numpy(), ".b", label = "target", markersize = 2, alpha = 0.6)
            plt.plot(X_test.data.numpy(), y_pred.data.numpy(), ".r", label = "initial", markersize = 2, alpha = 0.6)
            plt.plot(X_test.data.numpy(), y_pred_new.data.numpy(), ".y", label = "optimized", markersize = 2, alpha = 0.6)
            plt.plot(X_train.data.numpy(), y_train.data.numpy(), ".k", markersize = 5, label = "training points")
            plt.title("Only optimizing latent variable, validation")
            plt.legend()
            plt.show()

            master_model.get_statistics(X_train, y_train)
            master_model.use_clone_net(clone_parameters = True)
            y_pred = master_model(X_test)
            loss_list_1 = master_model.clone_net_quick_learn(X_train, y_train, validation_data = (X_test, y_test), batch_size = batch_size, 
                                                             epochs = epochs, lr = lr, optim_type = optim_type)
            y_pred_new = master_model(X_test)
            plt.plot(X_test.data.numpy(), y_test.data.numpy(), ".b", label = "target", markersize = 2, alpha = 0.6)
            plt.plot(X_test.data.numpy(), y_pred.data.numpy(), ".r", label = "initial", markersize = 2, alpha = 0.6)
            plt.plot(X_test.data.numpy(), y_pred_new.data.numpy(), ".y", label = "optimized", markersize = 2, alpha = 0.6)
            plt.plot(X_train.data.numpy(), y_train.data.numpy(), ".k", markersize = 5, label = "training points")
            plt.legend()
            plt.title("Optimizing cloned net, validation")
            plt.show()

            master_model.get_statistics(X_train, y_train)
            master_model.use_clone_net(clone_parameters = False)
            W_core, _ = master_model.cloned_net.get_weights_bias(W_source = "core")
            y_pred = master_model(X_test)
            loss_list_2 = master_model.clone_net_quick_learn(X_train, y_train, validation_data = (X_test, y_test), batch_size = batch_size, 
                                                             epochs = epochs, lr = lr, optim_type = optim_type)
            y_pred_new = master_model(X_test)
            plt.plot(X_test.data.numpy(), y_test.data.numpy(), ".b", label = "target", markersize = 3, alpha = 0.6)
            plt.plot(X_test.data.numpy(), y_pred.data.numpy(), ".r", label = "initial", markersize = 3, alpha = 0.6)
            plt.plot(X_test.data.numpy(), y_pred_new.data.numpy(), ".y", label = "optimized", markersize = 3, alpha = 0.6)
            plt.plot(X_train.data.numpy(), y_train.data.numpy(), ".k", markersize = 5, label = "training points")
            plt.xlabel("epochs")
            plt.ylabel("mse loss")
            plt.legend()
            plt.title("Optimizing from_scratch, validation")
            plt.show()

            plt.plot(range(len(loss_list_0)), loss_list_0, label = "loss_optim_latent_param")
            plt.plot(range(len(loss_list_1)), loss_list_1, label = "loss_clone")
            plt.plot(range(len(loss_list_2)), loss_list_2, label = "loss_scratch")
            plt.legend()
            plt.show()
            plt.semilogy(range(len(loss_list_0)), loss_list_0, label = "loss_optim_latent_param")
            plt.semilogy(range(len(loss_list_1)), loss_list_1, label = "loss_clone")
            plt.semilogy(range(len(loss_list_2)), loss_list_2, label = "loss_scratch")

            plt.legend()
            plt.show()

        elif plot_type == "slider":
            from matplotlib.widgets import Slider, Button, RadioButtons
            import matplotlib
            num_tasks_sampled = 10
            rand_id = sorted(np.random.randint(num_test_tasks, size = num_tasks_sampled))
            task_dict = {}
            sample_task_ids = []
            for i, idx in enumerate(rand_id):
                sample_task_id = task_id_list[0] + "_{0}".format(idx)
                sample_task_ids.append(sample_task_id)
                ((X_train, y_train), (X_test, y_test)), _ = tasks_test[sample_task_id]
                if i == 0:
                    X_test_0, y_test_0 = X_test, y_test
                    X_train_0, y_train_0 = X_train, y_train
                task_dict[sample_task_id] = [[X_train, y_train], [X_test, y_test]]
            X_train, y_train = X_train_0, y_train_0
            X_test, y_test = X_test_0, y_test_0

            if is_VAE:
                statistics = statistics_Net(torch.cat([X_train, y_train], 1))[0][0]
            else:
                statistics = statistics_Net(torch.cat([X_train, y_train], 1))[0]
            statistics_numpy = statistics.data.numpy()
            relevance = get_relevance(X_train, y_train, statistics_Net)
            pred = generative_Net(X_test, statistics)
            W_core, _ = generative_Net.get_weights_bias(W_source = "core")
            num_layers = len(W_core)

            fig, ax = plt.subplots(figsize = (9, 7.5))
            plt.subplots_adjust(left=0.25, bottom=0.27)
            ss0, ss1, ss2, ss3 = statistics_numpy.squeeze().tolist()
            subplt1 = plt.subplot2grid((3, num_layers), (0,0), rowspan = 2, colspan = num_layers)

            l_button_test, = subplt1.plot(X_test.data.numpy(), y_test.data.numpy(), ".r", markersize = 2, alpha = 0.6, label = "testing target")
            l_button, = subplt1.plot(X_train.data.numpy(), y_train.data.numpy(), ".k", markersize = 4, alpha = 0.6, label = "training point")
            # subplt1.scatter(X_train.data.numpy(), y_train.data.numpy(), s = relevance * 5 + 1, c = "k", alpha = 0.6)
            if is_uncertainty_net:
                statistics_logvar = statistics_Net(torch.cat([X_train, y_train], 1))[1]
                pred_std = torch.exp(master_model.generative_Net_logstd(X_test, statistics_logvar))
                l_errormean, _, l_errorbar = subplt1.errorbar(X_test.data.numpy().squeeze(), pred.data.numpy().squeeze(), yerr = pred_std.data.numpy().squeeze(), color = "b", fmt = ".", markersize = 1, alpha = 0.2)
            else:
                l, = subplt1.plot(X_test.data.numpy(), pred.data.numpy(),'.b', markersize = 2, alpha = 0.6, label = "predicted")
            subplt1.axis([task_settings["xlim"][0], task_settings["xlim"][1], -5, 5])
            subplt1.legend()

            subplt2s = []
            for i in range(num_layers):
                subplt2 = plt.subplot2grid((3, num_layers), (2, i))
                scale_min = np.floor(np.min(W_core[i]))
                scale_max = np.ceil(np.max(W_core[i]))
                subplt2.matshow(W_core[i], cmap = matplotlib.cm.binary, vmin = scale_min, vmax = scale_max)
                subplt2.set_xticks(np.array([]))
                subplt2.set_yticks(np.array([]))
                subplt2.set_xlabel("({0:.3f}, {1:.3f})".format(scale_min, scale_max), fontsize = 10)

            axcolor = 'lightgoldenrodyellow'
            ax0 = plt.axes([0.25, 0.08, 0.65, 0.025], facecolor=axcolor)
            ax1 = plt.axes([0.25, 0.12, 0.65, 0.025], facecolor=axcolor)
            ax2 = plt.axes([0.25, 0.16, 0.65, 0.025], facecolor=axcolor)
            ax3 = plt.axes([0.25, 0.2, 0.65, 0.025], facecolor=axcolor)

            s0 = Slider(ax0, 'Statistics[0]', -3, 3, valinit=ss0)
            s1 = Slider(ax1, 'Statistics[1]', -3, 3, valinit=ss1)
            s2 = Slider(ax2, 'Statistics[2]', -3, 3, valinit=ss2)
            s3 = Slider(ax3, 'Statistics[3]', -3, 3, valinit=ss3)

            def update(val):
                (X_train, y_train), (X_test, y_test) = task_dict[radio.value_selected]
                statistics_numpy = np.array([[s0.val, s1.val, s2.val, s3.val]])
                statistics = Variable(torch.FloatTensor(statistics_numpy))
                pred = generative_Net(X_test, statistics)
                l.set_ydata(pred.data.numpy())
                l.set_xdata(X_test.data.numpy())
                W_core, _ = generative_Net.get_weights_bias(W_source = "core")
                for i in range(num_layers):
                    subplt2 = plt.subplot2grid((3, num_layers), (2, i))
                    scale_min = np.floor(np.min(W_core[i]))
                    scale_max = np.ceil(np.max(W_core[i]))
                    subplt2.matshow(W_core[i], cmap = matplotlib.cm.binary, vmin = scale_min, vmax = scale_max)
                    subplt2.set_xticks(np.array([]))
                    subplt2.set_yticks(np.array([]))
                    subplt2.set_xlabel("({0:.3f}, {1:.3f})".format(scale_min, scale_max), fontsize = 10)
                fig.canvas.draw_idle()
            s0.on_changed(update)
            s1.on_changed(update)
            s2.on_changed(update)
            s3.on_changed(update)

            resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
            button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
            button.label.set_fontsize(8)

            def reset(event):
                s0.reset()
                s1.reset()
                s2.reset()
                s3.reset()
            button.on_clicked(reset)

            rax = plt.axes([0.025, 0.5 - 0.01 * num_tasks_sampled, 0.18, 0.03 * num_tasks_sampled], facecolor = axcolor)
            radio = RadioButtons(rax, sample_task_ids, active=0)

            def update_y_test(label):
                # Update the target_data:
                (X_train, y_train), (X_test, y_test) = task_dict[label]
                l_button.set_ydata(y_train.data.numpy())
                l_button.set_xdata(X_train.data.numpy())
                l_button_test.set_ydata(y_test.data.numpy())
                l_button_test.set_xdata(X_test.data.numpy())

                # Update the fitted data:
                if is_VAE:
                    statistics = statistics_Net(torch.cat([X_train, y_train], 1))[0][0]
                else:
                    statistics = statistics_Net(torch.cat([X_train, y_train], 1))[0]
                pred = generative_Net(X_test, statistics)
                if is_uncertainty_net:
                    statistics_logvar = statistics_Net(torch.cat([X_test, y_test], 1))[1]
                    pred_std = torch.exp(master_model.generative_Net_logstd(X_test, statistics_logvar))
                    l_errormean.set_xdata(X_test.data.numpy())
                    l_errormean.set_ydata(pred.data.numpy())
                    print(torch.cat([pred-pred_std, pred+pred_std], 1).data.numpy().shape)
                    l_errorbar[0].set_verts(torch.cat([pred-pred_std, pred+pred_std], 1).data.numpy())
                else:
                    l.set_xdata(X_test.data.numpy())
                    l.set_ydata(pred.data.numpy())
                    

                # Update the slider
                ss0, ss1, ss2, ss3 = statistics.data.numpy().squeeze().tolist()
                s0.set_val(ss0)
                s1.set_val(ss1)
                s2.set_val(ss2)
                s3.set_val(ss3)

                fig.canvas.draw_idle()
            radio.on_clicked(update_y_test)

            plt.show()
