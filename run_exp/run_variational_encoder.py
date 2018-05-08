import os, sys

exp_id=[
"enc1.0",
]

exp_mode = [
"meta",
# "baseline",
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
# '["bounce-states"]',
'["bounce-images"]',
]


statistics_output_neurons=[
# 4,
8,
# 10,
# 20,
]

is_VAE=[
# True,
False,
]

VAE_beta=[
# 1,
0.2,
# 2,
]

lr=[
# 5e-5,
2e-5,
]

batch_size_task=[
100,
]

pre_pooling_neurons=[
200,
400,
# 600,
]

num_context_neurons=[
0,
# 4,
# 8,
]

statistics_pooling=[
"max",
# "mean",
]

main_hidden_neurons=[
# '(40,40)',
'\(40,40,40\)',
# '(80,80)',
# '\(80,80,80\)',
]

reg_amp=[
# 1e-6,
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
"individual",
# "sum",
]

is_uncertainty_net = [
False,
# True,
]

loss_core=[
"huber",
# "mse",
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
            batch_size_task,
            pre_pooling_neurons,
            num_context_neurons,
            statistics_pooling,
            main_hidden_neurons,
            reg_amp,
            activation_gen,
            activation_model,
            optim_mode,
            is_uncertainty_net,
            loss_core,
]
param_chosen = assign_array_id(array_id, param_list)
exec_str = "python ../variational/variational_meta_learning_encoder.py"
for param in param_chosen:
    exec_str += " {0}".format(param)
exec_str += " {0}".format(array_id)
print(param_chosen)
print(exec_str)

os.system(exec_str)