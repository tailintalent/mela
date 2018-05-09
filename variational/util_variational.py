
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import time


# In[2]:


def process_object_info(percept_list, chosen_dim = None):
    """Transform the percepted list of dict structure into a plane dictionary structure."""
    # Currently it assumes that the objects will not disappear or reappear:
    perception_dict_whole = {}
    for percept_dict in percept_list:
        if len(percept_dict) == 0:
            for per_key, per_item in perception_dict_whole.items():
                per_item.append([np.NaN] * len(per_item[0]))
            perception_dict_whole
        for key, info_dict in percept_dict.items():
            for subkey, info in info_dict.items():
                idx = key + "_{0}".format(subkey)
                if idx not in perception_dict_whole:
                    perception_dict_whole[idx] = []
                perception_dict_whole[idx].append(info)
    for key in perception_dict_whole:
        perception_dict_whole[key] = np.array(perception_dict_whole[key])
        if chosen_dim is not None:
            chosen_dim = np.array(chosen_dim)
            perception_dict_whole[key] = perception_dict_whole[key][:, chosen_dim]
    return perception_dict_whole


def get_task(
    trajectory,
    bouncing_list,
    obs_array = None,
    time_steps = 3,
    forward_steps = 1,
    is_flatten = True,
    output_dims = None,
    isTorch = True,
    is_cuda = False,
    width = None,
    test_size = 0.2,
    ):  
    """Obtain training and testing data from a trajectory"""
    X = []
    y = []
    info = {}
    others = []
    reflect = []
    if obs_array is not None:
        obs_array = obs_array.squeeze()
        obs_array = obs_array
        obs_X = []
        obs_y = []
    
    # Configure forward steps:
    if not isinstance(forward_steps, list) and not isinstance(forward_steps, tuple):
        forward_steps = [forward_steps]
    forward_steps = list(forward_steps)
    assert min(forward_steps) == 1
    max_forward_steps = max(forward_steps)

    for i in range(len(trajectory) - time_steps - max_forward_steps + 1):
        X.append(trajectory[i: i + time_steps])
        y.append(trajectory[i + time_steps + np.array(forward_steps) - 1])
        reflect.append(np.any(bouncing_list[i + 1: i + time_steps + max_forward_steps]).astype(int))
        if obs_array is not None:
            obs_X.append(obs_array[i: i + time_steps])
            obs_y.append(obs_array[i + time_steps + np.array(forward_steps) - 1])
        if max_forward_steps > 1:
            others.append(trajectory[i + time_steps : i + time_steps + max_forward_steps - 1])
        else:
            others.append([1])
    X = np.array(X)
    y = np.array(y)
    reflect = np.array(reflect)
    others = np.array(others)
    if obs_array is not None:
        obs_X = np.array(obs_X)
        obs_y = np.array(obs_y)

    # Delete entries with NaN (indicating new game):
    valid_X = ~np.isnan(X.reshape(X.shape[0], -1).sum(1))
    valid_y = ~np.isnan(y.reshape(y.shape[0], -1).sum(1))
    valid_others = ~np.isnan(others.reshape(others.shape[0], -1).sum(1))
    valid = valid_X & valid_y & valid_others
    X = X[valid]
    y = y[valid]
    reflect = reflect[valid]
    if obs_array is not None:
        obs_X = obs_X[valid]
        obs_y = obs_y[valid]
        
        max_value = max(obs_X.max(), obs_y.max())
        obs_X = obs_X / max_value
        obs_y = obs_y / max_value
    
    if output_dims is not None:
        if not isinstance(output_dims, list) and not isinstance(output_dims, tuple):
            output_dims = [output_dims]
        output_dims = torch.LongTensor(np.array(output_dims))
        if is_cuda:
            output_dims = output_dims.cuda()
        y = y[..., output_dims]

    if is_flatten and obs_array is None:
        X = X.reshape(X.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        others = others.reshape(others.shape[0], -1)

    X_train, X_test, y_train, y_test, reflected_train, reflected_test = train_test_split(X, y, reflect, test_size = test_size)
    if obs_array is not None:
        obs_X_train, obs_X_test, obs_y_train, obs_y_test = train_test_split(obs_X, obs_y, test_size = test_size)
    input_size = X.shape[1:]
    output_size = y.shape[1:]
    info["input_size"] = input_size[0] if len(input_size) == 1 else input_size
    info["output_size"] = output_size[0] if len(output_size) == 1 else output_size
    if isTorch:
        X_train = Variable(torch.FloatTensor(X_train), requires_grad = False)
        y_train = Variable(torch.FloatTensor(y_train), requires_grad = False)
        X_test = Variable(torch.FloatTensor(X_test), requires_grad = False)
        y_test = Variable(torch.FloatTensor(y_test), requires_grad = False)
        reflected_train = Variable(torch.ByteTensor(reflected_train), requires_grad = False)
        reflected_test = Variable(torch.ByteTensor(reflected_test), requires_grad = False)
        if obs_array is not None:
            obs_X_train = Variable(torch.FloatTensor(obs_X_train), requires_grad = False)
            obs_y_train = Variable(torch.FloatTensor(obs_y_train), requires_grad = False)
            obs_X_test = Variable(torch.FloatTensor(obs_X_test), requires_grad = False)
            obs_y_test = Variable(torch.FloatTensor(obs_y_test), requires_grad = False)
        if is_cuda:
            X_train = X_train.cuda()
            y_train = y_train.cuda()
            X_test = X_test.cuda()
            y_test = y_test.cuda()
            reflected_train = reflected_train.cuda()
            reflected_test = reflected_test.cuda()
            if obs_array is not None:
                obs_X_train = obs_X_train.cuda()
                obs_y_train = obs_y_train.cuda()
                obs_X_test = obs_X_test.cuda()
                obs_y_test = obs_y_test.cuda()
    if obs_array is None:
        return ((X_train, y_train), (X_test, y_test), (reflected_train, reflected_test)), info
    else:
        info["states"] = ((X_train, y_train), (X_test, y_test))
        return ((obs_X_train, obs_y_train), (obs_X_test, obs_y_test), (reflected_train, reflected_test)), info


def get_env_data(
    env_name,
    data_format = "states",
    input_dims = 1,
    output_dims = None,
    ball_idx = 0,
    num_examples = 10000,
    test_size = 0.2,
    is_cuda = False,
    isplot = 1,
    verbose = True,
    **kwargs
    ):
    if env_name[:9] == "piecewise":
        env_name_split = env_name.split("-")
        input_size = int(env_name_split[1][:-1])
        num_pieces = int(env_name_split[2][:-1])
        num_boundaries = int(env_name_split[3][:-1])
        func_types = []
        for letter in env_name_split[4]:
            if letter == "l":
                func_types.append("linear")
            elif letter == "q":
                func_types.append("quadratic")
            else:
                raise Exception("letter {0} is not a valid function type!".format(letter))
        ((X_train, y_train), (X_test, y_test), (reflected_train, reflected_test)), info =             get_piecewise_dataset(input_size = input_size, 
                                  num_pieces = num_pieces,
                                  num_boundaries = num_boundaries,
                                  func_types = func_types,
                                  x_range = kwargs["x_range"] if "x_range" in kwargs else (0, 20),
                                  num_examples = num_examples,
                                  is_cuda = is_cuda,
                                  isplot = isplot,
                                 )
    
    else:
        from AI_scientist.settings.a2c_env_settings import ENV_SETTINGS_CHOICE
        from AI_scientist.variational.envs import make_env
        from AI_scientist.util import plot_matrices
        import random

        # Obtain settings from kwargs:
        time_steps = kwargs["time_steps"] if "time_steps" in kwargs else 3
        forward_steps = kwargs["forward_steps"] if "forward_steps" in kwargs else 1
        episode_length = kwargs["episode_length"] if "episode_length" in kwargs else 100
        is_flatten = kwargs["is_flatten"] if "is_flatten" in kwargs else False
        render = kwargs["render"] if "render" in kwargs else False
        env_name_split = env_name.split("-")
        if "nobounce" in env_name_split:
            env_name_core = "-".join(env_name_split[:-1])
        else:
            env_name_core = env_name
        env_settings = {key: random.choice(value) if isinstance(value, list) else value for key, value in ENV_SETTINGS_CHOICE[env_name_core].items()}
        env_settings["info_contents"] = ["coordinates"]
        max_distance = env_settings["max_distance"] if "max_distance" in env_settings else None
        max_range = env_settings["max_range"] if "max_range" in env_settings else None
        if max_range is not None:
            value_min, value_max = max_range
        input_dims = env_settings["input_dims"] if "input_dims" in env_settings else input_dims

        # Reset certain aspects of the environment:
        if "screen_width" in kwargs:
            print("corrected screen_width: {0}".format(kwargs["screen_width"]))
            env_settings["screen_width"] = kwargs["screen_width"]
        if "screen_height" in kwargs:
            print("corrected screen_height: {0}".format(kwargs["screen_height"]))
            env_settings["screen_height"] = kwargs["screen_height"]
        if "physics" in kwargs:
            print("corrected physics: {0}".format(kwargs["physics"]))
            env_settings["physics"] = kwargs["physics"]
        if "boundaries" in kwargs:
            print("corrected boundaries: {0}".format(kwargs["boundaries"]))
            env_settings["boundaries"] = kwargs["boundaries"]
        if "ball_vmax" in kwargs:
            print("corrected ball_vmax: {0}".format(kwargs["ball_vmax"]))
            env_settings["ball_vmax"] = kwargs["ball_vmax"]
        if "step_dt" in kwargs:
            print("corrected step_dt: {0}".format(kwargs["step_dt"]))
            env_settings["step_dt"] = kwargs["step_dt"]
        env = make_env("Breakout_Custom-v0", 1, 0, "", clip_rewards = False, env_settings = env_settings)()
        env.allow_early_resets = True

        obs_var = []
        info_list = []
        bouncing_list = []

        k = 0
        while k < int(num_examples / episode_length):
            obs = env.reset()
            obs_var_candidate = []
            info_list_candidate = []
            bouncing_list_candidate = []
            ball_x = None
            ball_y = None
            is_break = False
            # Obtain the frames:
            for i in range(episode_length):
                obs, _, _, info = env.step(1)
                obs_var_candidate.append(obs)
                coordinates = info["coordinates"]
                info_list_candidate.append(coordinates)
                bouncing_list_candidate.append(info["ball_bouncing_info"])
                if max_distance is not None:
                    last_ball_x = ball_x
                    last_ball_y = ball_y
                    ball_x, ball_y = coordinates["ball"][ball_idx]
                    if last_ball_x is not None:
                        if abs(ball_x - last_ball_x) > max_distance:
                            is_break = True
                            if verbose:
                                print("{0} break for too large velocity.".format(k))
                            break
                if max_range is not None:
                    ball_x, ball_y = coordinates["ball"][ball_idx]
                    if ball_x < value_min or ball_x > value_max or ball_y < value_min or ball_y > value_max:
                        is_break = True
                        if verbose:
                            print("{0} break for going outsize the max_range".format(k))
                        break
                if render:
                    time.sleep(0.1)
                    env.render('human')
            # Only add the episode if it is does not break:
            if not is_break:
                obs_var = obs_var + obs_var_candidate
                info_list = info_list + info_list_candidate
                bouncing_list = bouncing_list + bouncing_list_candidate
                obs_var.append({})
                info_list.append({})
                bouncing_list.append({})
                k += 1
        if isplot > 0:
            plot_matrices(np.array(obs_var[:30]).squeeze())

        # Process the info_list into numpy format:
        perception_dict = process_object_info(info_list, chosen_dim = input_dims)
        bouncing_list = [element[ball_idx][0] if len(element) > 0 else np.NaN for element in bouncing_list]
        if data_format == "images":
            obs_array = np.array([element if len(element) > 0 else np.full(obs_var[0].shape, np.nan) for element in obs_var])
        else:
            obs_array = None
        trajectory0 = perception_dict["ball_{0}".format(ball_idx)]
        width = env_settings["screen_width"] if input_dims == 0 else env_settings["screen_height"]
        ((X_train, y_train), (X_test, y_test), (reflected_train, reflected_test)), info =             get_task(trajectory0,
                     bouncing_list = bouncing_list,
                     obs_array = obs_array,
                     time_steps = time_steps,
                     forward_steps = forward_steps,
                     is_flatten = is_flatten,
                     output_dims = output_dims,
                     is_cuda = is_cuda,
                     width = width,
                     test_size = test_size
                    )

        if "nobounce" in env_name_split:
            if obs_array is None:
                X_train = X_train[reflected_train.unsqueeze(1) == 0].view(-1, X_train.size(1))
                y_train = y_train[reflected_train.unsqueeze(1) == 0].view(-1, y_train.size(1))
                X_test = X_test[reflected_test.unsqueeze(1) == 0].view(-1, X_test.size(1))
                y_test = y_test[reflected_test.unsqueeze(1) == 0].view(-1, y_test.size(1))
            else:
                X_train = X_train[reflected_train.view(reflected_train.size(0), 1,1,1) == 0].view(-1, *X_train.size()[1:])
                y_train = y_train[reflected_train.view(reflected_train.size(0), 1,1,1) == 0].view(-1, *y_train.size()[1:])
                X_test = X_test[reflected_test.view(reflected_test.size(0), 1,1,1) == 0].view(-1, *X_test.size()[1:])
                y_test = y_test[reflected_test.view(reflected_test.size(0), 1,1,1) == 0].view(-1, *y_test.size()[1:])
            reflected_train = reflected_train[reflected_train == 0]
            reflected_test = reflected_test[reflected_test == 0]
    return ((X_train, y_train), (X_test, y_test), (reflected_train, reflected_test)), info


def get_torch_tasks(tasks, task_id, num_tasks = None, is_cuda = False):
    tasks_dict = OrderedDict()
    for i, task in enumerate(tasks):
        if num_tasks is not None and i > num_tasks:
            break
        ((X_train_numpy, y_train_numpy), (X_test_numpy, y_test_numpy)), z_info = task
        X_train = Variable(torch.FloatTensor(X_train_numpy), requires_grad = False)
        y_train = Variable(torch.FloatTensor(y_train_numpy), requires_grad = False)
        X_test = Variable(torch.FloatTensor(X_test_numpy), requires_grad = False)
        y_test = Variable(torch.FloatTensor(y_test_numpy), requires_grad = False)
        if is_cuda:
            X_train = X_train.cuda()
            y_train = y_train.cuda()
            X_test = X_test.cuda()
            y_test = y_test.cuda()
        tasks_dict["{0}_{1}".format(task_id, i)] = [[[X_train, y_train], [X_test, y_test]], z_info]
    return tasks_dict


def get_numpy_tasks(tasks):
    tasks_save = []
    for task_key, task in tasks.items():
        ((X_train, y_train), (X_test, y_test)), z_info = task
        tasks_save.append([[[X_train.data.numpy(), y_train.data.numpy()], [X_test.data.numpy(), y_test.data.numpy()]], z_info])
    return tasks_save

