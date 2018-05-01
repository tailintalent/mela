import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

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


def get_task(trajectory, bouncing_list, time_steps = 3, forward_steps = 1, is_flatten = True, isTorch = True, is_cuda = False, width = None):
    """Obtain training and testing data from a trajectory"""
    X = []
    y = []
    info = {}
    others = []
    reflect = []
    for i in range(len(trajectory) - time_steps - forward_steps + 1):
        X.append(trajectory[i: i + time_steps])
        y.append(trajectory[i + time_steps + forward_steps - 1: i + time_steps + forward_steps])
        reflect.append(np.any(bouncing_list[i + 1: i + time_steps + forward_steps]).astype(int))
        if forward_steps > 1:
            others.append(trajectory[i + time_steps : i + time_steps + forward_steps - 1])
        else:
            others.append([1])
    X = np.array(X)
    y = np.array(y)
    reflect = np.array(reflect)
    # reflect = np.array(bouncing_list)[time_steps:]
    others = np.array(others)
    if is_flatten:
        X = X.reshape(X.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        others = others.reshape(others.shape[0], -1)

    # Delete entries with NaN (indicating new game):
    valid_X = ~np.isnan(X.reshape(X.shape[0], -1).sum(1))
    valid_y = ~np.isnan(y.reshape(y.shape[0], -1).sum(1))
    valid_others = ~np.isnan(others.reshape(others.shape[0], -1).sum(1))
    valid = valid_X & valid_y & valid_others
    X = X[valid]
    y = y[valid]
    reflect = reflect[valid]

    X_train, X_test, y_train, y_test, reflected_train, reflected_test = train_test_split(X, y, reflect, test_size = 0.2)
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
        if is_cuda:
            X_train = X_train.cuda()
            y_train = y_train.cuda()
            X_test = X_test.cuda()
            y_test = y_test.cuda()
            reflected_train = reflected_train.cuda()
            reflected_test = reflected_test.cuda()
    return ((X_train, y_train), (X_test, y_test), (reflected_train, reflected_test)), info


def get_env_data(
    env_name,
    input_dims = 1,
    output_dims = 0,
    ball_idx = 0,
    num_examples = 10000,
    is_cuda = False,
    isplot = 1,
    render = False,
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
        ((X_train, y_train), (X_test, y_test), (reflected_train, reflected_test)), info = \
            get_piecewise_dataset(input_size = input_size, 
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
        time_steps = kwargs["time_steps"] if "time_steps" in kwargs else 3
        forward_steps = kwargs["forward_steps"] if "forward_steps" in kwargs else 1
        episode_length = kwargs["episode_length"] if "episode_length" in kwargs else 100
        env_name_split = env_name.split("-")
        if "nobounce" in env_name_split:
            env_name_core = "-".join(env_name_split[:-1])
        else:
            env_name_core = env_name
        env_settings = {key: random.choice(value) if isinstance(value, list) else value for key, value in ENV_SETTINGS_CHOICE[env_name_core].items()}
        env_settings["info_contents"] = ["coordinates"]
        max_distance = env_settings["max_distance"] if "max_distance" in env_settings else None
        input_dims = env_settings["input_dims"] if "input_dims" in env_settings else input_dims

        if "screen_width" in kwargs:
            print("corrected screen_width: {0}".format(kwargs["screen_width"]))
            env_settings["screen_width"] = kwargs["screen_width"]
        if "screen_height" in kwargs:
            print("corrected screen_height: {0}".format(kwargs["screen_height"]))
            env_settings["screen_height"] = kwargs["screen_height"]
        if "physics" in kwargs:
            print("corrected physics: {0}".format(kwargs["physics"]))
            env_settings["physics"] = kwargs["physics"]
        if "ball_vmax" in kwargs:
            print("corrected ball_vmax: {0}".format(kwargs["ball_vmax"]))
            env_settings["ball_vmax"] = kwargs["ball_vmax"]
        env = make_env("Breakout_Custom-v0", 1, 0, "", clip_rewards = False, env_settings = env_settings)()
        env.allow_early_resets = True
        print(env_settings)

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
                if render:
                    time.sleep(0.1)
                    env.render('human')
            if not is_break:
                obs_var = obs_var + obs_var_candidate
                info_list = info_list + info_list_candidate
                bouncing_list = bouncing_list + bouncing_list_candidate
                info_list.append({})
                bouncing_list.append({})
                k += 1
        obs_var = np.array(obs_var)
        obs_var = Variable(torch.FloatTensor(obs_var), requires_grad = False)

        if isplot > 0:
            plot_matrices(obs_var[:30,0,...].data.numpy())

        perception_dict = process_object_info(info_list, chosen_dim = input_dims)
        bouncing_list = [element[ball_idx][0] if len(element) > 0 else np.NaN for element in bouncing_list]
        trajectory0 = perception_dict["ball_{0}".format(ball_idx)]
        width = env_settings["screen_width"] if input_dims == 0 else env_settings["screen_height"]
        ((X_train, y_train), (X_test, y_test), (reflected_train, reflected_test)), info = \
            get_task(trajectory0,
                     bouncing_list = bouncing_list,
                     time_steps = time_steps,
                     forward_steps = forward_steps,
                     is_cuda = is_cuda,
                     width = width,
                    )
        if output_dims is not None:
            if not isinstance(output_dims, list):
                output_dims = [output_dims]
            output_dims = torch.LongTensor(np.array(output_dims))
            if is_cuda:
                output_dims = output_dims.cuda()
            y_train = y_train[:, output_dims]
            y_test = y_test[:, output_dims]
        if "nobounce" in env_name_split:
            # reflected_train = ((torch.abs((X_train[:,0] + X_train[:,2] - 2 * X_train[:,1])) > 1e-5) | (torch.abs((X_train[:,1] + y_train[:, 0] - 2 * X_train[:,2])) > 1e-5)).long()
            # reflected_test = ((torch.abs((X_test[:,0] + X_test[:,2] - 2 * X_test[:,1])) > 1e-5) | (torch.abs((X_test[:,1] + y_test[:, 0] - 2 * X_test[:,2])) > 1e-5)).long()
            
            X_train = X_train[reflected_train.unsqueeze(1) == 0].view(-1, X_train.size(1))
            y_train = y_train[reflected_train.unsqueeze(1) == 0].view(-1, y_train.size(1))
            X_test = X_test[reflected_test.unsqueeze(1) == 0].view(-1, X_test.size(1))
            y_test = y_test[reflected_test.unsqueeze(1) == 0].view(-1, y_test.size(1))
            reflected_train = reflected_train[reflected_train == 0]
            reflected_test = reflected_test[reflected_test == 0]
    return ((X_train, y_train), (X_test, y_test), (reflected_train, reflected_test)), info