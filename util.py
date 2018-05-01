from __future__ import print_function
import numpy as np
from copy import deepcopy
import os


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    sample_size = len(labels_dense)
    labels_one_hot = np.zeros((sample_size, num_classes))
    labels_one_hot[np.arange(sample_size), np.array(labels_dense).astype(int)] = 1
    return labels_one_hot.astype(int)


def save_h5weights(model,filename='network.h5'):
    import h5py
    W_list, b_list = model.get_weights_bias()
    h5f = h5py.File(filename,'w')
    for i in range(0,len(W_list)):
        h5f.create_dataset("W"+str(1+i), data=W_list[i])
    for i in range(0,len(b_list)):
        h5f.create_dataset("b"+str(i), data=b_list[i])
    h5f.close()
    return


def plot_matrices(
    matrix_list, 
    shape = None, 
    images_per_row = 10, 
    scale_limit = None,
    figsize = (20, 8), 
    x_axis_list = None,
    filename = None,
    title = None,
    highlight_bad_values = True,
    plt = None,
    pdf = None,
    ):
    """Plot the images for each matrix in the matrix_list."""
    import matplotlib
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize = figsize)
    fig.set_canvas(plt.gcf().canvas)
    if title is not None:
        fig.suptitle(title, fontsize = 18, horizontalalignment = 'left', x=0.1)
    
    num_matrixs = len(matrix_list)
    rows = np.ceil(num_matrixs / float(images_per_row))
    try:
        matrix_list_reshaped = np.reshape(np.array(matrix_list), (-1, shape[0],shape[1])) \
            if shape is not None else np.array(matrix_list)
    except:
        matrix_list_reshaped = matrix_list
    if scale_limit == "auto":
        scale_min = np.Inf
        scale_max = -np.Inf
        for matrix in matrix_list:
            scale_min = min(scale_min, np.min(matrix))
            scale_max = max(scale_max, np.max(matrix))
        scale_limit = (scale_min, scale_max)
    for i in range(len(matrix_list)):
        ax = fig.add_subplot(rows, images_per_row, i + 1)
        image = matrix_list_reshaped[i].astype(float)
        if len(image.shape) == 1:
            image = np.expand_dims(image, 1)
        if highlight_bad_values:
            cmap = matplotlib.cm.binary
            cmap.set_bad('red', alpha = 0.2)
            mask_key = []
            mask_key.append(np.isnan(image))
            mask_key.append(np.isinf(image))
            mask_key = np.any(np.array(mask_key), axis = 0)
            image = np.ma.array(image, mask = mask_key)
        else:
            cmap = matplotlib.cm.binary
        if scale_limit is None:
            ax.matshow(image, cmap = cmap)
        else:
            assert len(scale_limit) == 2, "scale_limit should be a 2-tuple!"
            ax.matshow(image, cmap = cmap, vmin = scale_limit[0], vmax = scale_limit[1])
        try:
            xlabel = "({0:.4f},{1:.4f})\nshape: ({2}, {3})".format(np.min(image), np.max(image), image.shape[0], image.shape[1])
            if x_axis_list is not None:
                xlabel += "\n{0}".format(x_axis_list[i])
            plt.xlabel(xlabel)
        except:
            pass
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    
    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename)
    if pdf is not None:
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
    else:
        plt.show()

    if scale_limit is not None:
        print("scale_limit: ({0:.6f}, {1:.6f})".format(scale_limit[0], scale_limit[1]))
    print()


class Gradient_Noise_Scale_Gen(object):
    def __init__(
        self,
        gamma = 0.55,
        eta = 0.01,
        noise_scale_start = 1e-2,
        noise_scale_end = 1e-6,
        gradient_noise_interval_batch = 1,
        fun_pointer = "generate_scale_simple",
        batch_size = 50,
        ):
        self.gamma = gamma
        self.eta = eta
        self.noise_scale_start = noise_scale_start
        self.noise_scale_end = noise_scale_end
        self.gradient_noise_interval_batch = gradient_noise_interval_batch
        self.batch_size = batch_size
        self.generate_scale = getattr(self, fun_pointer) # Sets the default function to generate scale
        
    def get_max_iter(self, epochs, num_examples):
        self.epochs = epochs
        self.num_examples = num_examples
        self.max_iter = int(self.epochs * self.num_examples / self.batch_size / self.gradient_noise_interval_batch) + 1
    
    def generate_scale_simple(
        self,
        epochs,
        num_examples,
        verbose = True
        ):
        self.get_max_iter(epochs, num_examples)       
        gradient_noise_scale = np.sqrt(self.eta * (np.array(range(self.max_iter)) + 1) ** (- self.gamma))
        if verbose:
            print("gradient_noise_scale: start = {0}, end = {1:.6f}, gamma = {2}, length = {3}".format(gradient_noise_scale[0], gradient_noise_scale[-1], self.gamma, self.max_iter))
        return gradient_noise_scale

    def generate_scale_fix_ends(
        self,
        epochs,
        num_examples,
        verbose = True,
        ):
        self.get_max_iter(epochs, num_examples)
        ratio = (self.noise_scale_start / float(self.noise_scale_end)) ** (1 / self.gamma) - 1
        self.bb = self.max_iter / ratio
        self.aa = self.noise_scale_start * self.bb ** self.gamma
        gradient_noise_scale = np.sqrt(self.aa * (np.array(range(self.max_iter)) + self.bb) ** (- self.gamma))
        if verbose:
            print("gradient_noise_scale: start = {0}, end = {1:.6f}, gamma = {2}, length = {3}".format(gradient_noise_scale[0], gradient_noise_scale[-1], self.gamma, self.max_iter))
        return gradient_noise_scale


def plot_pdf(input_, sigma_value, plot_threshold = 0.001):
    """Plot the density function of the weights.
    The input_ can either be a tuple of (x_axis, density) or a single weight tensor
    """
    from matplotlib import pyplot as plt
    import tensorflow as tf
    if isinstance(input_, tuple) and len(input_) == 2:
        x_axis, density_tensor = input_
        density = density_tensor.eval({sigma: sigma_value})
        if plot_threshold is not None and plot_threshold > 0:
            for i in range(len(x_axis)):
                if density[i] > plot_threshold:
                    start = i
                    break
            for i in range(len(x_axis) - 1, 0, -1):
                if density[i] > plot_threshold:
                    end = i
                    break
            x_axis = x_axis[start: end]
            density = density[start: end]
        plt.plot(x_axis, density)
    else:
        weight = input_
        def get_mixed_Gaussian(weight, x, sigma_value):
            """helper function to calculate the integrand value at specific x"""
            weight_flatten = tf.reshape(weight, [tf.size(weight).eval()]).eval()
            out = (np.sum(np.exp( - (x - weight_flatten) ** 2 / (2 * sigma_value ** 2)))) ** 2
            return out
    
        value_min = np.min(weight.eval())
        value_max = np.max(weight.eval())
        x_axis = np.linspace(value_min - 2 * sigma_value, value_max + 2 * sigma_value, 100)
        y_axis = []
        for x in x_axis:
            y_axis.append(get_mixed_Gaussian(weight, x, sigma_value))
        plt.plot(x_axis, y_axis)
    plt.show()


def plot_density(input_, sigma = None, x_label = None, y_label = None, xlim = None):
    from scipy.stats import gaussian_kde
    from matplotlib import pyplot as plt
    density = gaussian_kde(input_)
    xs = np.linspace(np.min(input_), np.max(input_), 400)
    if sigma is None:
        sigma = (np.max(input_) - np.max(input_)) / 100
    density.covariance_factor = lambda : sigma
    density._compute_covariance()
    plt.plot(xs, density(xs))
    if xlim is not None:
        plt.xlim(xlim)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.show()


def record_weights(weight_record_list, weights_to_reg, chosen_index = None):
    """transform the weight tensor into a numpy array and save as the same list structure as weights_to_reg"""
    if len(weight_record_list) == 0:
        for weights in weights_to_reg:
            if isinstance(weights, list):
                length = len(weights)
            else:
                length = 1
            weight_record_list.append([[] for i in range(length)])
                             
    for i in range(len(weights_to_reg)):
        weights = weights_to_reg[i]
        if isinstance(weights, list):
            for j in range(len(weights)):
                weight = weights[j]
                if chosen_index is None:
                    weight_record_list[i][j].append(np.ndarray.flatten(weight.eval()))
                else:
                    weight_record_list[i][j].append(np.ndarray.flatten(weight.eval())[chosen_index])
        else:
            if chosen_index is None:
                weight_record_list[i][0].append(np.ndarray.flatten(weights.eval()))
            else:
                weight_record_list[i][0].append(np.ndarray.flatten(weights.eval())[chosen_index])


def record_info(info_record_list, info_list, feed_dict = {}, tf_Tensor = None):
    """Record the information into info_record_list"""
    if tf_Tensor is None:
        import tensorflow as tf
        tf_Tensor = tf.Tensor
    info_record = []
    for info in info_list:
        if isinstance(info, list):
            info_ele_list = []
            for info_ele in info:
                if isinstance(info_ele, tf_Tensor):
                    info_ele_list.append(info_ele.eval(feed_dict = feed_dict))
                else:
                    info_ele_list.append(info_ele)
            info_record.append(info_ele_list)
        else:
            if isinstance(info, tf_Tensor):
                info_record.append(info.eval(feed_dict = feed_dict))
            else:
                info_record.append(info)
    info_record_list.append(info_record)


def decompose_list(input_):
    """Recusively decompose any list structure into a flat list"""
    def dec(input_, output_):
        if type(input_) is list:
            for subitem in input_:
                dec(subitem, output_)
        else:
            output_.append(input_)
    output_ = []
    dec(input_, output_)
    return output_


def get_dir(filename):
    """Get the full directory path. If not exist, create one."""
    current_directory = os.path.dirname(os.path.realpath(__file__))
    # Create the directory if it does not exist:
    index = filename.rfind("/")
    dir_name = os.path.join(current_directory, filename[:index])
    isExist = os.path.isdir(dir_name)
    if not isExist:
        os.makedirs(dir_name)
    return os.path.join(current_directory, filename)


def make_dir(filename):
    import os
    import errno
    if not os.path.exists(os.path.dirname(filename)):
        print("directory {0} does not exist, created.".format(os.path.dirname(filename)))
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                print(exc)
                raise


def print_struct_param(struct_param, transition_mode = "layertype-only", print_mode = "long"):
    """Print the struct_param in a concise way."""
    if print_mode == "long":
        struct_param_new = []
        for layer_struct_param in struct_param:
            num_neurons, layer_mode, layer_hyperparam = layer_struct_param
            struct_param_new.append([num_neurons, layer_hyperparam["weight"]["type"], layer_hyperparam["bias"]["type"]])
        return struct_param_new

    elif print_mode == "short":
        if transition_mode == "num-neurons-only":
            return [layer_struct_param[0] for layer_struct_param in struct_param]
        elif transition_mode == "layertype-only":
            return [[layer_struct_param[2]["weight"]["type"], layer_struct_param[2]["bias"]["type"]] 
                        for layer_struct_param in struct_param]
        else:
            struct_param_new = []
            for layer_struct_param in struct_param:
                num_neurons, layer_mode, layer_hyperparam = layer_struct_param
                struct_param_new.append([num_neurons, layer_hyperparam["weight"]["type"], layer_hyperparam["bias"]["type"]])
            return struct_param_new
    else:
        raise Exception("print_mode must be either 'long' or 'short'!")


def rotate_matrix_cw(matrix, angle = 90):
    """Rotate the matrix clockwise by certain angle (multiples of 90 deg)."""
    def rotate_matrix_90(matrix):
        rows, columns = matrix.shape
        matrix_new = np.zeros((columns, rows))
        for i in range(rows):
            for j in range(columns):
                matrix_new[j, rows - 1 - i] = matrix[i, j]
        return matrix_new

    assert isinstance(angle, int) and angle % 90 == 0, "The rotation angle must be multiples of 90 deg!"
    times = (angle % 360) / 90
    for k in range(times):
        matrix = rotate_matrix_90(matrix)
    return matrix


def record_data(data_record_dict, data_list, key_list):
    """Record data to the dictionary data_record_dict. It records each key: value pair in the corresponding location of 
    key_list and data_list into the dictionary."""
    assert len(data_list) == len(key_list), "the data_list and key_list should have the same length!"
    for data, key in zip(data_list, key_list):
        if key not in data_record_dict:
            data_record_dict[key] = [data]
        else: 
            data_record_dict[key].append(data)


def sort_two_lists(list1, list2, reverse = False):
    from operator import itemgetter
    if reverse:
        List = deepcopy([list(x) for x in zip(*sorted(zip(deepcopy(list1), deepcopy(list2)), key=itemgetter(0), reverse=True))])
    else:
        List = deepcopy([list(x) for x in zip(*sorted(zip(deepcopy(list1), deepcopy(list2)), key=itemgetter(0)))])
    if len(List) == 0:
        return [], []
    else:
        return List[0], List[1]


def get_new_name(name_prev):
    List = name_prev.split("_")
    try:
        suffix = str(eval(List[-1]) + 1)
        name = List[:-1] + [suffix]
    except:
        suffix = "0"
        name = List + [suffix]
    name = "_".join(name)
    return name


def truncated_normal(shape, init_mean, init_std):
    """Truncated normal function, where the examples that are outside of 2 init_std are thrown out."""
    from scipy.stats import truncnorm
    sample = truncnorm.rvs(-2, 2, size = shape)
    return sample * init_std + init_mean


def add_scaled_noise_to_gradients(grads_and_vars, gradient_noise_scale):
    """Adds scaled noise from a 0-mean normal distribution to gradients."""
    import tensorflow as tf
    from tensorflow.python.framework import ops
    gradients, variables = zip(*grads_and_vars)
    noisy_gradients = []
    for gradient in gradients:
        if gradient is None:
            noisy_gradients.append(None)
            continue
        if isinstance(gradient, ops.IndexedSlices):
            gradient_shape = gradient.dense_shape
        else:
            gradient_shape = gradient.get_shape()
        noise = tf.truncated_normal(gradient_shape) * gradient_noise_scale
        noisy_gradients.append(gradient + noise)
    return list(zip(noisy_gradients, variables))


def plot_record(model_param, key_list = ["loss_train", "loss_valid", "reg_S_entropy", "reg_L1", "reg_L1_selector"], log_scale = False):
    import matplotlib.pyplot as plt
    if isinstance(model_param, dict):
        data_record = model_param["data_record"]
    else:
        data_record = model_param.data_record
    record_list = {}
    for key in key_list:
        if key not in data_record:
            continue
        record_list[key] = data_record[key]
        plt.plot(data_record["epoch"], record_list[key], label = key)
    if log_scale:
        plt.yscale('log')
    plt.legend()
    plt.show()


def softmax(X, axis = -1):
    X_max = np.amax(X, axis, keepdims = True)
    X = np.exp(X - X_max)
    return X / X.sum(axis = axis, keepdims = True)


def manifold_embedding(X, color = None, all_methods = None):
    from matplotlib import pylab as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.ticker import NullFormatter
    from sklearn import manifold
    from time import time
    if color is None:
        color = np.ones(X.shape[0])
    elif color == "linspace":
        color = np.linspace(0, 1, X.shape[0])
    if all_methods is None:
        all_methods = ['standard', 'ltsa', 'hessian', 'modified', "Isomap", "MDS", "SpectralEmbedding", "t-SNE"]

    # Next line to silence pyflakes. This import is needed.
    Axes3D

    n_points = len(X)
    n_neighbors = 10
    n_components = 2
    marker_size = 1
    cmap_scale = (np.min(color), np.max(color))

    fig = plt.figure(figsize=(20, 15))
    plt.suptitle("Manifold Learning with %i points, %i neighbors"
                 % (n_points, n_neighbors), fontsize=14)

    ax = fig.add_subplot(251, projection='3d')
    cax = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral, s = marker_size, vmin= cmap_scale[0], vmax=cmap_scale[1])
    ax.view_init(4, -72)
    cbar = fig.colorbar(cax, ticks=[cmap_scale[0], np.mean(cmap_scale), cmap_scale[1]]) # color bar
    cbar.ax.set_yticklabels(['{0:.3f}'.format(cmap_scale[0]), '{0:.3f}'.format(np.mean(cmap_scale)), '{0:.3f}'.format(cmap_scale[1])])

    methods = ['standard', 'ltsa', 'hessian', 'modified']
    labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

    for i, method in enumerate(methods):
        if method in all_methods:
            try:
                t0 = time()
                Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                                    eigen_solver='auto',
                                                    method=method).fit_transform(X)
                t1 = time()
                print("%s: %.2g sec" % (methods[i], t1 - t0))

                ax = fig.add_subplot(252 + i)
                plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, s = marker_size, vmin= cmap_scale[0], vmax=cmap_scale[1])
                plt.title("%s (%.2g sec)" % (labels[i], t1 - t0))
                ax.xaxis.set_major_formatter(NullFormatter())
                ax.yaxis.set_major_formatter(NullFormatter())
                plt.axis('tight')
            except:
                print("method {0} failed!".format(method))

    if "Isomap" in all_methods:
        t0 = time()
        Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
        t1 = time()
        print("Isomap: %.2g sec" % (t1 - t0))
        ax = fig.add_subplot(257)
        plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, s = marker_size, vmin= cmap_scale[0], vmax=cmap_scale[1])
        plt.title("Isomap (%.2g sec)" % (t1 - t0))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')

    if "MDS" in all_methods:
        t0 = time()
        mds = manifold.MDS(n_components, max_iter=100, n_init=1)
        Y = mds.fit_transform(X)
        t1 = time()
        print("MDS: %.2g sec" % (t1 - t0))
        ax = fig.add_subplot(258)
        plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, s = marker_size, vmin= cmap_scale[0], vmax=cmap_scale[1])
        plt.title("MDS (%.2g sec)" % (t1 - t0))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')

    if "SpectralEmbedding" in all_methods:
        t0 = time()
        se = manifold.SpectralEmbedding(n_components=n_components,
                                        n_neighbors=n_neighbors)
        Y = se.fit_transform(X)
        t1 = time()
        print("SpectralEmbedding: %.2g sec" % (t1 - t0))
        ax = fig.add_subplot(259)
        plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, s = marker_size, vmin= cmap_scale[0], vmax=cmap_scale[1])
        plt.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')

    if "t-SNE" in all_methods:
        t0 = time()
        tsne = manifold.TSNE(n_components=n_components, perplexity = 30, init='pca', random_state=0)
        Y = tsne.fit_transform(X)
        t1 = time()
        print("t-SNE: %.2g sec" % (t1 - t0))
        ax = fig.add_subplot(2, 5, 10)
        plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, s = marker_size, vmin= cmap_scale[0], vmax=cmap_scale[1])
        plt.title("t-SNE (%.2g sec)" % (t1 - t0))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')

    plt.show()


def get_struct_str(struct_param):
    def get_struct_str_ele(struct_param):
        return "-".join(["{0}{1}".format(struct_param[k][0], struct_param[k][1][:2]) for k in range(len(struct_param))])
    if isinstance(struct_param, tuple):
        return ",".join([get_struct_str_ele(struct_param_ele) for struct_param_ele in struct_param])
    else:
        return get_struct_str_ele(struct_param)


def get_args(arg, arg_id = 1, type = "str"):
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
        arg_return = arg
    except:
        import sys
        try:
            arg_return = sys.argv[arg_id]
            if type == "int":
                arg_return = int(arg_return)
            elif type == "float":
                arg_return = float(arg_return)
            elif type == "bool":
                arg_return = eval(arg_return)
            elif type == "eval":
                arg_return = eval(arg_return)
            elif type == "tuple":
                splitted = arg_return[1:-1].split(",")
                List = []
                for item in splitted:
                    try:
                        item = eval(item)
                    except:
                        pass
                    List.append(item)
                arg_return = tuple(List)
            elif type == "str":
                pass
            else:
                raise Exception("type {0} not recognized!".format(type))
        except:
            raise
            arg_return = arg
    return arg_return


class Early_Stopping(object):
    def __init__(self, patience = 100, epsilon = 0, mode = "min"):
        self.patience = patience
        self.epsilon = epsilon
        self.mode = "min"
        self.best_value = None
        self.wait = 0
        
    def monitor(self, value):
        to_stop = False
        if self.patience is not None:
            if self.best_value is None:
                self.best_value = value
                self.wait = 0
            else:
                if (self.mode == "min" and value < self.best_value - self.epsilon) or \
                   (self.mode == "max" and value > self.best_value + self.epsilon):
                    self.best_value = value
                    self.wait = 0
                else:
                    if self.wait >= self.patience:
                        to_stop = True
                    else:
                        self.wait += 1
        return to_stop


    def reset(self):
        self.best_value = None
        self.wait = 0


def get_highlight_fun(highlight_columns = None, mode = "min"):
    """For pandas dataframe, highlighting the min/max values in a column"""
    def highlight(s):
        if mode == "min":
            if highlight_columns is None:
                chosen = (s == s.min())
            else:
                chosen = (s == s.min()) & (s.name in highlight_columns)
        elif mode == "max":
            if highlight_columns is None:
                chosen = (s == s.max())
            else:
                chosen = (s == s.max()) & (s.name in highlight_columns)
        return ['background-color: darkorange' if v else '' for v in chosen]
    return highlight


def get_int_str(start, end):
    string = ""
    for i in range(start, end + 1):
        string += "{0} ".format(i)
    return string


def new_dict(Dict, new_content_dict):
    new_Dict = deepcopy(Dict)
    new_Dict.update(new_content_dict)
    return new_Dict


def base_repr(n, base, length):
    assert n < base ** length, "n should be smaller than b ** length"
    base_repr_str = np.base_repr(n, base, padding = length)[-length:]
    return [int(ele) for ele in base_repr_str]


def base_repr_2_int(List, base):
    if len(List) == 1:
        return List[0]
    elif len(List) == 0:
        return 0
    else:
        return base * base_repr_2_int(List[:-1], base) + List[-1]