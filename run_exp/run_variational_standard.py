import os, sys

exp_id=[
"syn1.02",
]

exp_mode = [
"meta",
#"finetune",
#"oracle",
]

task_id_list=[
# '["latent-linear"]',
# '["polynomial-3"]',
# '["Legendre-3"]',
# '["M-sawtooth"]',
# '["M-sin"]',
# '["M-Gaussian"]',
# '["M-tanh"]',
# '["M-softplus"]',
'["C-sin"]',
'["C-tanh"]',
# '["bounce-states"]',
# '["bounce-images"]',
]


statistics_output_neurons=[
2,
4,
# 8,
# 10,
# 20,
]

is_VAE=[
True,
False,
]

VAE_beta=[
# 1,
0.2,
# 2,
]

lr=[
2e-4,
1e-4,
5e-5,
#2e-5,
]

pre_pooling_neurons=[
200,
#300,
# 600,
]

num_context_neurons=[
0,
4,
# 8,
]

statistics_pooling=[
"max",
# "mean",
]

struct_param_pre_neurons=[
'\(60,3\)',
#'\(60,2\)',
]

struct_param_gen_base_neurons=[
#'\(60,3\)',
'\(30,2\)',
'\(60,2\)',
'\(120,2\)',
]

main_hidden_neurons=[
'\(40,40\)',
# '\(40,40,40\)',
# '(80,80)',
# '\(80,80,80\)',
]

reg_amp=[
#1e-5,
1e-6,
# 0,
]

activation_gen=[
"leakyRelu",
# "elu",
]

activation_model=[
"leakyRelu",
# "elu",
]

optim_mode=[
"indi",
#"sum",
]

is_uncertainty_net = [
False,
# True,
]

loss_core=[
"huber",
# "mse",
]

patience=[
200,
300,
]

def assign_array_id(array_id, param_list):
    if len(param_list) == 0:
        print("redundancy: {0}".format(array_id))
        return []
    else:
        param_bottom = param_list[-1]
        length = len(param_bottom)
        current_param = param_bottom[array_id % length]
        return assign_array_id(int(array_id / length), param_list[:-1]) + [current_param]

array_id = int(sys.argv[1])

param_list = [exp_id,
            exp_mode,
            task_id_list,
            statistics_output_neurons,
            is_VAE,
            VAE_beta,
            lr,
            pre_pooling_neurons,
            num_context_neurons,
            statistics_pooling,
            struct_param_pre_neurons,
            struct_param_gen_base_neurons,
            main_hidden_neurons,
            reg_amp,
            activation_gen,
            activation_model,
            optim_mode,
            is_uncertainty_net,
            loss_core,
            patience,
]
param_chosen = assign_array_id(array_id, param_list)
exec_str = "python ../variational/Experiments/variational_meta_learning_standard.py"
for param in param_chosen:
    exec_str += " {0}".format(param)
exec_str += " {0}".format(array_id)
print(param_chosen)
print(exec_str)

os.system(exec_str)
