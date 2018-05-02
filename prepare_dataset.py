from __future__ import print_function
from sklearn.datasets import fetch_mldata
import numpy as np
from numpy.random import choice


def Dataset_Gen(dataset_key, settings = {}):
	from sklearn.model_selection import train_test_split
	info_dict = {}
	if dataset_key[:9]  == "benchmark":
		(X_train, y_train), (X_test, y_test) = get_data_from_benchmark(mode = dataset_key[10:], one_hot = True)
		input_size = X_train.shape[1]
		output_size = y_train.shape[1]
		info_dict["loss_type"] = "cross-entropy"
		info_dict["struct_string"] = "100 7 4 1 100 7 4 1 100 7 4 1 {0} 7 4 1".format(output_size)
		activation_available = ["linear", "relu"]
		info_dict["bias_type_available"] = ["None", "constant", "dense"]
		epochs = 600
		if dataset_key == "benchmark_MNIST":
			info_dict["shape"] = (28, 28)

	elif dataset_key[:6] == "memory":
		memory = pickle.load(open("../{0}/{1}".format(settings["dataset_PATH"], dataset), 'rb'))
		screens = memory[0]
		input_size = 84 * 84
		X = np.reshape(screens, [-1, input_size])
		info_dict["dataset_str"] = dataset_key
	elif dataset_key[:4] == "auto":
		# Autoencoder experiment:
		epochs = 200
		activation_available = None
		num_examples = settings["num_examples"] if "num_examples" in settings else 20000
		if dataset_key == "auto_paddle":
			height = settings["height"]
			width = settings["width"]
			paddle_length = settings["paddle_length"]
			input_size = height * width
			X = np.zeros((num_examples, height, width))
			for i in range(num_examples):
				pos = np.random.randint(width - paddle_length + 1)
				X[i, :, pos: pos + paddle_length] = 1
			info_dict["dataset_str"] = dataset_key + "_{0}x{1}_len{2}".format(height, width, paddle_length)
		elif dataset_key == "auto_ball":
			height = settings["height"]
			width = settings["width"]
			input_size = height * width
			ball_size = settings["ball_size"]
			X = np.zeros((num_examples, height, width))
			for k in range(num_examples):
				ii = np.random.randint(height - ball_size + 1)
				jj = np.random.randint(width - ball_size + 1)
				X[k, ii : ii + ball_size, jj : jj + ball_size] = 1
			info_dict["dataset_str"] = dataset_key + "_{0}x{1}_size{2}".format(height, width, ball_size)
		elif dataset_key == "auto_ball_multiframe":
			height = settings["height"]
			width = settings["width"]
			ball_size = settings["ball_size"]
			steps = settings["steps"]
			v_max = settings["v_max"]
			input_size = height * width * steps
			X = np.zeros((num_examples, steps, height, width))

			for k in range(num_examples):
				ii = np.random.randint(height - ball_size + 1)
				jj = np.random.randint(width - ball_size + 1)
				v_ii = np.random.randint(-v_max, v_max + 1)
				v_jj = np.random.randint(-v_max, v_max + 1)
				for ll in range(steps):
					p_ii = ii + v_ii * ll
					p_jj = jj + v_jj * ll
					if p_ii < 0:
						p_ii = - p_ii
					elif p_ii >= width - ball_size:
						p_ii = 2 * (width - ball_size) - p_ii
					if p_jj < 0:
						p_jj = - p_jj
					elif p_jj >= height - ball_size:
						p_jj = 2 * (height - ball_size) - p_jj
					X[k, ll, p_ii : p_ii + ball_size, p_jj : p_jj + ball_size] = 1
			info_dict["dataset_str"] = dataset_key + "_{0}x{1}_size{2}_step{3}_vmax{4}".format(height, width, ball_size, steps, v_max)
		elif dataset_key == "auto_ball_multiframe_xy":
			height = settings["height"]
			width = settings["width"]
			ball_size = settings["ball_size"]
			steps = settings["steps"]
			v_max = settings["v_max"]
			input_size = steps * 2
			X = np.zeros((num_examples, steps, 2))
			for k in range(num_examples):
				ii = np.random.randint(height - ball_size + 1)
				jj = np.random.randint(width - ball_size + 1)
				v_ii = np.random.randint(-v_max, v_max + 1)
				v_jj = np.random.randint(-v_max, v_max + 1)
				for ll in range(steps):
					p_ii = ii + v_ii * ll
					p_jj = jj + v_jj * ll
					if p_ii < 0:
						p_ii = - p_ii
					elif p_ii >= width - ball_size:
						p_ii = 2 * (width - ball_size) - p_ii
					if p_jj < 0:
						p_jj = - p_jj
					elif p_jj >= height - ball_size:
						p_jj = 2 * (height - ball_size) - p_jj
					X[k, ll, :] = np.array([p_ii, p_jj])
			info_dict["dataset_str"] = dataset_key + "_{0}x{1}_size{2}_step{3}_vmax{4}_xy".format(height, width, ball_size, steps, v_max)
		info_dict["shape"] = X.shape[1:]
		y = X
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
	

	elif dataset_key == "sawtooth":
		input_size = 1
		(X_train, y_train), (X_test, y_test) = get_periodic_wave(mode = "triangular",
																 period = 2, 
																 height = 1, 
																 domain = settings["domain"] if "domain" in settings else (-5, 5), 
																 num_train = settings["num_train"] if "num_train" in settings else 200,
																 num_test = settings["num_test"] if "num_test" in settings else 200,
																)
		activation_available = ["linear", "relu"]
		epochs = 300

	elif dataset_key == "sin":
		input_size = 1
		(X_train, y_train), (X_test, y_test) = get_periodic_wave(mode = "sin",
																 period = 2,
																 height = 1,
																 domain = settings["domain"] if "domain" in settings else (-5, 5), 
																 num_train = settings["num_train"] if "num_train" in settings else 200,
																 num_test = settings["num_test"] if "num_test" in settings else 200,
																)
		activation_available = ["linear", "relu"]
		epochs = 300

	elif dataset_key[:8] == "Legendre":
		input_size = 1
		(X_train, y_train), (X_test, y_test) = get_polynomial(mode = "Legendre",
															  order = int(dataset_key.split("_")[1]),
															  domain = settings["domain"] if "domain" in settings else (-1, 1), 
															  num_train = settings["num_train"] if "num_train" in settings else 200,
															  num_test = settings["num_test"] if "num_test" in settings else 200,
															 )
		activation_available = ["linear", "relu"]
		epochs = 300
		
	elif dataset_key[:3] == "xor":
		input_size = 20
		(X_train, y_train), (X_test, y_test) = get_logical(input_size = input_size,
														   mode = "xor",
														   num_train = settings["num_train"] if "num_train" in settings else 5000,
														   num_test = settings["num_test"] if "num_test" in settings else 2000,
														   one_hot = False,
														  )
		epochs = 4000
		if dataset_key[3:] == "Relu" or dataset_key[3:] == "":
			activation_available = ["linear", "relu"]
		elif dataset_key[3:] == "Sin":
			activation_available = ["linear", "sin"]
	elif "or" in dataset_key or "and" in dataset_key:
		activation_available = ["linear"]
		if "Relu" in dataset_key:
			activation_available.append("relu")
		elif "Sigmoid" in dataset_key:
			activation_available.append("sigmoid")
		elif "Tanh" in dataset_key:
			activation_available.append("tanh")
		else:
			activation_available += ["relu", "sigmoid", "tanh"]
		
		input_size = 20
		(X_train, y_train), (X_test, y_test) = get_logical(input_size = input_size, mode = "or" if "or" in dataset_key else "and", 
														   num_train = settings["num_train"] if "num_train" in settings else 5000,
														   num_test = settings["num_test"] if "num_test" in settings else 2000,
														   one_hot = False,
														  )
		epochs = 2000

	elif dataset_key == "rolling":
		input_size = 8
		(X_train, y_train), (X_test, y_test) = get_rolling_sequence(input_size = input_size,
																	num_train = settings["num_train"] if "num_train" in settings else 5000,
																	num_test = settings["num_test"] if "num_test" in settings else 2000,
														  		   )
		epochs = 1000
		activation_available = ["linear", "relu"]
	else:
		raise Exception("dataset {0} not valid!".format(dataset_key))
	
	# Processing dataset and info_dict:
	if "isTorch" in settings and settings["isTorch"]:
		import torch
		from torch.autograd import Variable
		X_train = Variable(torch.FloatTensor(X_train), requires_grad = False)
		y_train = Variable(torch.FloatTensor(y_train), requires_grad = False)
		X_test = Variable(torch.FloatTensor(X_test), requires_grad = False)
		y_test = Variable(torch.FloatTensor(y_test), requires_grad = False)
	dataset = ((X_train, y_train), (X_test, y_test))
	epochs = settings["epochs"] if "epochs" in settings else epochs
	info_dict["input_size"] = input_size
	info_dict["activation_available"] = activation_available
	info_dict["epochs"] = epochs
	if "dataset_str" not in info_dict:
		info_dict["dataset_str"] = dataset_key

	return dataset, info_dict


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    sample_size = len(labels_dense)
    labels_one_hot = np.zeros((sample_size, num_classes))
    labels_one_hot[np.arange(sample_size), np.array(labels_dense).astype(int)] = 1
    return labels_one_hot.astype(int)


def get_data_from_csv(filename, X_columns, y_columns):
    """Loading the data from csv"""
    from sklearn.model_selection import train_test_split
    data = np.genfromtxt(filename, delimiter=',')
    if isinstance(X_columns, int):
        X_start = X_end = X_columns
    else:
        assert len(X_columns) == 2, "X_columns must be an int or a 2-element array/tuple!"
        X_start, X_end = X_columns
    if isinstance(y_columns, int):
        y_start = y_end = y_columns
    else:
        assert len(y_columns) == 2, "y_columns must be an int or a 2-element array/tuple!"
        y_start, y_end = y_columns
        
    X = data[:, X_start: X_end + 1]
    y = data[:, y_start: y_end + 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    return (X_train, y_train), (X_test, y_test)


def get_data_from_benchmark(mode, one_hot = True):
	"""Get data from Bengio's ICML 2007 paper "An Empirical Evaluation of Deep Architectures
	   on Problems with Many Factors of Variation"
	"""

	if mode == "MNIST":
		(X_train, y_train), (X_test, y_test) = get_MNIST(mode = "original", digit_per_class = None, one_hot = False)
		X_train = np.reshape(X_train, (-1, 784))
		X_test = np.reshape(X_test, (-1, 784))
		num_classes = 10
	else:
		if mode == "MNIST-BG-IMG-ROT":
			filename = "datasets/MNIST-BG-IMG-ROT/MNIST-BG-IMG-ROT_{0}.amat"
			num_classes = 10
		elif mode == "RECT":
			filename = "datasets/rectangles/rectangles_{0}.amat"
			num_classes = 2
		elif mode == "CONVEX":
			filename = "datasets/convex/convex_{0}.amat"
			num_classes = 2
		data_train = np.genfromtxt(filename.format("train"))
		data_test = np.genfromtxt(filename.format("test"))
		X_train = data_train[:,:-1]
		y_train = data_train[:,-1].astype(int)
		X_test = data_test[:,:-1]
		y_test = data_test[:,-1].astype(int)
	if one_hot:
		y_train = dense_to_one_hot(y_train, num_classes = num_classes)
		y_test = dense_to_one_hot(y_test, num_classes = num_classes)
	else:
		y_train = np.expand_dims(y_train, 1)
		y_test = np.expand_dims(y_test, 1)
	return (X_train, y_train), (X_test, y_test)


def OR(x):
    return float(sum(x)>0.5)

def XOR(x):
    return float(sum(x)%2)

def get_OR(
	input_size = 6,
	one_hot = True,
	plus_minus_format = False,
	):
	"""
	Generate the data whose label is an OR function of all possible combinations of input bits.
	
	Parameters
	----------
	input_size : int
	  Number of input bits.
	  
	one_hot: bool
	  Whether the labels y_train and y_test are in one_hot format. Default True.
	
	plus_minus_format : bool
      Use 1 and -1 instead of 1 and 0 as binary encoding. Default False.
	"""
	
	n = 2**input_size

	def bits(i):
		return np.unpackbits(np.uint8(i))[-input_size:]

	X = np.array(map(bits,range(n))).astype('int')
	y = np.array(map(OR,X)).astype('int')
	if plus_minus_format:
		X = 2 * X - 1
	if one_hot:
		y = dense_to_one_hot(y, 2)
	else:
		y = np.expand_dims(y, 1)
	return (X, y)

def get_XOR(
	input_size = 6,
	one_hot = True,
	plus_minus_format = False,
	):
	"""
	Generate the data whose label is an OR function of all possible combinations of input bits.
	
	Parameters
	----------
	input_size : int
	  Number of input bits.
	  
	one_hot: bool
	  Whether the labels y_train and y_test are in one_hot format. Default True.
	
	plus_minus_format : bool
      Use 1 and -1 instead of 1 and 0 as binary encoding. Default False.
	"""
	
	n = 2**input_size

	def bits(i):
		return np.unpackbits(np.uint8(i))[-input_size:]

	X = np.array(map(bits,range(n))).astype('int')
	y = np.array(map(XOR,X)).astype('int')
	if plus_minus_format:
		X = 2 * X - 1
	if one_hot:
		y = dense_to_one_hot(y, 2)
	else:
		y = np.expand_dims(y, 1)
	return (X, y)


def get_logical(
    input_size = 10,
    mode = "and",  # Choose from "and", "or" or "xor"
    num_train = 1000,
    num_test = 1000,
    one_hot = False,    
    ):
    from sklearn.model_selection import train_test_split
    num_examples = num_train + num_test

    data_examples = []
    for i in range(num_examples):
        num_ones = choice(input_size + 1)
        ones_pos = choice(input_size, size = num_ones, replace = False)

        data_example = np.zeros(input_size)
        data_example[ones_pos] = 1
        data_examples.append(data_example.tolist())
    data_examples = np.array(data_examples)

    if mode == "and":
        data_labels = (np.sum(data_examples, 1) == input_size).astype(int)
    elif mode == "or":
        data_labels = (np.sum(data_examples, 1) >= 1).astype(int)
    elif mode == "xor":
    	data_labels = (np.sum(data_examples, 1) % 2).astype(int)

    # Process one_hot:
    if one_hot:
        data_labels = dense_to_one_hot(data_labels, 2)
    else:
        data_labels = np.expand_dims(data_labels, 1)

    X_train, X_test, y_train, y_test = train_test_split(data_examples, data_labels, test_size= num_test / float(num_train + num_test))
    return (X_train, y_train), (X_test, y_test)


def get_xor_binomial(
	input_size = 6,
	positive_proba_train = 0.5,
	positive_proba_test = 0.5,
	num_train = 100,
	num_test = 100,
	one_hot = True,
	plus_minus_format = False,
	):
	"""
	Generate the data whose label is an XOR function of the binary input
	
	Parameters
	----------
	input_size : int
	  Size of the binary input.
	  
	positive_proba_train : float
	  Probability of each element of training example being 1.
	  
	positive_proba_test : float
	  Probability of each element of testing example being 1.
	  
	num_train : int
	  Number of training data.
	  
	num_test : int
	  Number of testing data.

	one_hot: bool
	  Whether the labels y_train and y_test are in one_hot format. Default True.

	plus_minus_format : bool
      Use 1 and -1 instead of 1 and 0 as binary encoding. Default False.
	"""
	assert 0 <= positive_proba_train <= 1, "positive_proba_train must be inside [0,1]!"
	assert 0 <= positive_proba_test <= 1, "positive_proba_test must be inside [0,1]!"
	X_train = choice(2, size = [num_train, input_size], p = [1- positive_proba_train, positive_proba_train])
	X_test = choice(2, size = [num_test, input_size], p = [1- positive_proba_test, positive_proba_test])
	y_train = np.sum(X_train, 1) % 2
	y_test = np.sum(X_test, 1) % 2
	if plus_minus_format:
		X_train = 2 * X_train - 1
		X_test = 2 * X_test - 1
	if one_hot:
		y_train = dense_to_one_hot(y_train, 2)
		y_test = dense_to_one_hot(y_test, 2)
	else:
		y_train = np.expand_dims(y_train, 1)
		y_test = np.expand_dims(y_test, 1)
	return (X_train, y_train), (X_test, y_test)


def get_periodic_wave(
	mode = "triangular",
	period = 2,
	height = 1,
	domain = (0, 10),
	num_train = 100,
	num_test = 100,
	noise_std = 0,
	random_phase = True,
	):
	"""
	Generate the periodic wave data with certain Gaussian noise (with standard deviation sigma)
	
	Parameters
	----------
	mode : str
	  type of the periodic wave. Choose between "triangular", "sin".
	
	period : float (or int)
	  Peirod of the tranglewave function
	  
	height : float (or int)
	  Height of the tranglewave function
	  
	domain : (float, float)
	  Domain of the tranglewave function. The X_train and X_test are sampled uniformly in the domain
	  
	num_train : int
	  Number of training data.
	  
	num_test : int
	  Number of testing data.
	
	noise_std : float
	  The standard deviation of the Gaussian noise. Default 0 meaning no noise.
	"""
	def trianglewave(
		x,
		period = 2,
		height = 1,
		):
		"""
		Generates the tranglewave function
		"""
		remainder = x % period
		slope = height / float(period) * 2
		return np.minimum(slope * remainder, 2 * height - slope * remainder)

	if random_phase:
		phase = np.random.rand()
	else:
		phase = 0
	
	# Generates the x uniform
	X_train = np.random.rand(num_train) * (domain[1] - domain[0]) + domain[0]
	X_test = np.random.rand(num_test) * (domain[1] - domain[0]) + domain[0]
	if mode == "triangular":
		y_train = trianglewave(X_train + phase * period, period = period, height = height)
		y_test = trianglewave(X_test + phase * period, period = period, height = height)
	elif mode == "sin":
		y_train = np.sin((X_train / period + phase) * 2 * np.pi) * height
		y_test = np.sin((X_test / period + phase) * 2 * np.pi) * height

	if noise_std > 0:
		y_train = y_train + noise_std * np.random.randn(num_train)
		y_test = y_test + noise_std * np.random.randn(num_test)
	X_train = np.expand_dims(X_train, 1)
	X_test = np.expand_dims(X_test, 1)
	y_train = np.expand_dims(y_train, 1)
	y_test = np.expand_dims(y_test, 1)
		
	return (X_train, y_train), (X_test, y_test)
		

def get_polynomial(
	mode = "Legendre",
	order = 2,
	domain = (-3, 3),
	num_train = 100,
	num_test = 100,
	noise_std = 0,
	):
	"""Get polynomials of different type and order

	Parameters
	----------
	mode : str
	  type of the periodic wave. Choose between "Legendre".

	order : int
	  Order of the polynormial.
	  
	num_train : int
	  Number of training data.
	  
	num_test : int
	  Number of testing data.

	noise_std : float
	  The standard deviation of the Gaussian noise. Default 0 meaning no noise.
	"""
	X_train = np.random.rand(num_train) * (domain[1] - domain[0]) + domain[0]
	X_test = np.random.rand(num_test) * (domain[1] - domain[0]) + domain[0]
	if mode == "Legendre":
		c = [0] * order + [1]
		y_train = np.polynomial.legendre.legval(X_train, c)
		y_test = np.polynomial.legendre.legval(X_test, c)
	if noise_std > 0:
		y_train = y_train + noise_std * np.random.randn(num_train)
		y_test = y_test + noise_std * np.random.randn(num_test)
	X_train = np.expand_dims(X_train, 1)
	X_test = np.expand_dims(X_test, 1)
	y_train = np.expand_dims(y_train, 1)
	y_test = np.expand_dims(y_test, 1)
	return (X_train, y_train), (X_test, y_test)


def get_binary_sequence(
    code = "10101",
    input_size = 10,
    num_train = 100,
    num_test = 100,
    one_hot = True,
    plus_minus_format = True,
    ):
    """
    Generates 1D bianry binary sequences with label y=1 only if it contains the given code. The sequence has 
    periodic boundary conditions.

    Parameters
    ----------
    code : str
      The code given. The label y = 1 if and only if the binary sequence contains the code given.

    input_size : int
      size of the image that the code can be translated on.

    num_train : int
      Number of training data.

    num_test : int
      Number of testing data.

    one_hot: bool
      Whether the labels y_train and y_test are in one_hot format. Default True.

    plus_minus_format : bool
      Use 1 and -1 instead of 1 and 0 as binary encoding. Default True.
    """
    assert len(code) <= input_size, "The length of hidden code should not exceed the input"
    from sklearn.model_selection import train_test_split
    data_examples = []
    data_labels = []
    for i in range(num_train + num_test):
        integer = np.random.randint(2 ** input_size)
        binary = np.binary_repr(integer, width = input_size)
        label = 0
        for i in range(input_size):
            binary_shift = binary[i:] + binary[:i]
            if code in binary_shift:
                is_code = True
                label = 1
                break
        if plus_minus_format:
        	binary = (np.array([2 * int(letter) - 1 for letter in binary])).astype(int)
        else:
        	binary = (np.array([int(letter) for letter in binary])).astype(int)
        data_examples.append(binary)
        data_labels.append(label)
    data_examples = np.array(data_examples)
    data_labels = np.array(data_labels)
    if one_hot:
        data_labels = dense_to_one_hot(data_labels, 2)
    else:
        data_labels = np.expand_dims(data_labels, 1)
    X_train, X_test, y_train, y_test = train_test_split(data_examples, data_labels, test_size= num_test / float(num_train + num_test))
    return (X_train, y_train), (X_test, y_test)


def get_rolling_sequence(
	code = "101011",
	input_size = 10,
	num_train = 100,
	num_test = 100,
	one_hot = True,
	):
	"""
	Generates 1D rolling binary sequences with label y indicating how much the original sequence has been rolled.
	The sequence has periodic boundary conditions.

	Parameters
	----------
	code : str
		The code given. The label y = 1 if and only if the binary sequence contains the code given.

	input_size : int
		size of the image that the code can be translated on.

	num_train : int
		Number of training data.

	num_test : int
		Number of testing data.

	one_hot: bool
		Whether the labels y_train and y_test are in one_hot format. Default True.
	"""
	assert len(code) <= input_size, "The length of code should not exceed the input"
	from sklearn.model_selection import train_test_split
	num_examples = num_train + num_test  
	sequence = np.pad(np.array([eval(element) for element in list(code)]), [0, input_size - len(code)], "constant")
	full_data = np.repeat(np.expand_dims(sequence,0), num_examples, axis = 0)
	data_labels = choice(range(input_size), size = num_examples) # Each code is randomly translated on the image.
	data_examples = np.array([np.roll(full_data[i], data_labels[i], axis = 0) for i in range(num_examples)])

	if one_hot:
		data_labels = dense_to_one_hot(data_labels, num_classes = input_size)
	else:
		data_labels = np.expand_dims(data_labels, 1)
		
	X_train, X_test, y_train, y_test = train_test_split(data_examples, data_labels, test_size= num_test / float(num_train + num_test))
	return (X_train, y_train), (X_test, y_test)



def get_and_or_binomial(
    input_size = 10,
    num_train = 1000,
    num_test = 1000,
    fraction_single_example = 0.5, #fraction of label 1 (for and) and label 0 (for or)
    one_hot = True,
    mode = "and",  # Choose from "and" and "or"
    ):
    from sklearn.model_selection import train_test_split
    # Get examples:
    if fraction_single_example is None:
    	data_examples = np.random.randint(2, size = (num_train + num_test, input_size))
    else:
    	from random import shuffle
    	data_examples1 = np.random.randint(2, size = (int((num_train + num_test) * (1 - fraction_single_example + 2 ** (-input_size))), 
    												 input_size))
    	if mode == "and":
    		data_examples2 =  np.ones((int((num_train + num_test) * (fraction_single_example + 2 ** (-input_size))), input_size))
    	elif mode == "or":
    		data_examples2 = np.zeros((int((num_train + num_test) * (fraction_single_example + 2 ** (-input_size))), input_size))
    	else:
    		raise Exception("mode {0} not recognized!".format(mode))
    	data_examples = np.concatenate((data_examples1, data_examples2))

    # Get labels:
    if mode == "and":
    	data_labels = (np.sum(data_examples, 1) == input_size).astype(int)
    elif mode == "or":
    	data_labels = (np.sum(data_examples, 1) >= 1).astype(int)

    # Process one_hot:
    if one_hot:
        data_labels = dense_to_one_hot(data_labels, 2)
    else:
        data_labels = np.expand_dims(data_labels, 1)

    X_train, X_test, y_train, y_test = train_test_split(data_examples, data_labels, test_size= num_test / float(num_train + num_test))
    return (X_train, y_train), (X_test, y_test)

	
def get_1D_code(
	code_size = 5,
	num_labels = 4,
	image_size = 10,
	num_train = 100,
	num_test = 100,
	one_hot = True,
	):
	"""
	Generates 1D bianry codes with translational symmetry. Each code has "1" at the left-most
	and right-most positions, so that we can identify the starting and ending of the code when it is translated.
	For example, 11001 is a valid code. 10100 is not a valid code.
	
	Parameters
	----------
	code_size : int
	  The size of each code. For example, 11001 has code_size of 5. Since all codes has "1" at the left-most
	and right-most positions, the effective length of the code is code_size - 2. The num_labels cannot exceed
	2 ** (code_size - 2).
	  
	num_labels : int
	  The number of different codes. It cannot exceed 2 ** (code_size - 2).
	  
	image_size : int
	  size of the image that the code can be translated on.
	
	num_train : int
	  Number of training data.
	  
	num_test : int
	  Number of testing data.

	one_hot: bool
	  Whether the labels y_train and y_test are in one_hot format. Default True.
	"""
	assert code_size > 2, "code size should be larger than 2!"
	assert image_size >= code_size, "image_size should be equal to or larger than code_size!"
	assert num_labels <= 2 ** (code_size - 2), "number_labels should not exceed 2 ^ (code_size - 2)"
	from sklearn.model_selection import train_test_split
	chosen_codes = sorted(choice(2 ** (code_size - 2), size = num_labels, replace = False))
	chosen_codes_binary = []
	white_space = image_size - code_size
	# Generate num_lables different binary codes with size of code_size:
	for i in range(num_labels):
		label = i
		binary = np.binary_repr(chosen_codes[i], width = code_size - 2)
		binary = np.concatenate((np.array([1] + [int(letter) for letter in binary] + [1]), np.zeros(white_space))).astype(int)
		chosen_codes_binary.append((binary, label))
		print(binary[:code_size], "label: {0}".format(label))

	# Construct the training and testing sets.
	data_examples = []
	data_labels = []
	for i in range(num_train + num_test):
		idx = choice(num_labels)
		example, label = chosen_codes_binary[idx]
		shift_amount = choice(white_space + 1) # Each code is randomly translated on the image.
		example = np.roll(example, shift_amount)
		data_examples.append(example)
		data_labels.append(label)
	data_examples = np.array(data_examples)
	data_labels = np.array(data_labels)
	if one_hot:
		data_labels = dense_to_one_hot(data_labels, num_labels)
	else:
		data_labels = np.expand_dims(data_labels, 1)

	X_train, X_test, y_train, y_test = train_test_split(data_examples, data_labels, test_size= num_test / float(num_train + num_test))
	return (X_train, y_train), (X_test, y_test)


def get_1D_double_code(
	code_size = 5,
	num_labels = 4,
	code_margin = 2,
	image_size = 12,
	num_train = 100,
	num_test = 100,
	one_hot = False,
	):
	"""
	Generates 1D double bianry codes with translational symmetry. For example, a double code
	may look like 010011000111010000, where the there are two single codes "10011" and "11101",
	separated by a margin of 3. Each single binary code has "1" at the left-most and right-most 
	positions, so that we can identify the starting and ending of the code when it is translated.

	
	Parameters
	----------
	code_size : int
	  The size of each code. For example, 11001 has code_size of 5. Since all codes has "1" at the left-most
	and right-most positions, the effective length of the code is code_size - 2. The num_labels cannot exceed
	2 ** (code_size - 2).
	  
	num_labels : int
	  The number of different codes. It cannot exceed 2 ** (code_size - 2).

	code_margin: int
	  The margin between the two single-codes in the double code.
	  
	image_size : int
	  size of the image that the code can be translated on.
	
	num_train : int
	  Number of training data.
	  
	num_test : int
	  Number of testing data.

	one_hot: bool
	  Whether the labels y_train and y_test are in one_hot format
	"""
	assert code_size > 2, "code size should be larger than 2!"
	assert image_size >= 2 * code_size + code_margin, "image_size should be equal to or larger than (2 * code_size + code_margin)!"
	assert num_labels <= 2 ** (code_size - 2), "number_labels should not exceed 2 ^ (code_size - 2)"
	from sklearn.model_selection import train_test_split
	chosen_codes = sorted(choice(2 ** (code_size - 2), size = num_labels, replace = False))
	chosen_codes_binary = []
	white_space = image_size - (2 * code_size + code_margin)
	# Generate num_lables different binary codes with size of code_size:
	for i in range(num_labels):
		label = i
		binary = np.binary_repr(chosen_codes[i], width = code_size - 2)
		binary = (np.array([1] + [int(letter) for letter in binary] + [1])).astype(int)
		chosen_codes_binary.append((binary, label))
		print(binary, "label: {0}".format(label))

	# Construct the training and testing sets.
	data_examples = []
	data_labels = []
	for i in range(num_train + num_test):
		idx1, idx2 = choice(num_labels, size = 2)
		code1, label1 = chosen_codes_binary[idx1]
		code2, label2 = chosen_codes_binary[idx2]
		label = (label1, label2) # Since it is double code, it has double labels.
		code = np.concatenate((code1, np.zeros(code_margin), code2, np.zeros(white_space))).astype(int)
		shift_amount = choice(white_space + 1) # Each code is randomly translated on the image.
		code = np.roll(code, shift_amount)
		data_examples.append(code)
		data_labels.append(label)
	if one_hot:
		data_labels = dense_to_one_hot(data_labels, num_labels)
	else:
		data_labels = np.expand_dims(data_labels, 1)
	X_train, X_test, y_train, y_test = train_test_split(data_examples, data_labels, test_size= num_test / float(num_train + num_test))
	return (X_train, y_train), (X_test, y_test)



def get_MNIST(
	mode = "random translation",
	core_size = (22, 22), 
	target_image_size = (28, 28),
	digit_per_class = None,
	one_hot = False,
	):
	"""This function fetches the original MNIST dataset.
	Both the X_train and X_test have the shape of (-1, 28, 28).

	Parameters
	----------
	X : np.array
	  Input feature matrix (N, row, column), 3D numpy array
	
	mode : str
	  The method to present the training and test image. Choose from the following modes:
	  "original" : return the training and testing images intact.
	  "random translation": return the training and testing images with random translation, 
	  		within the boundary of the image
	  "random translation torus": return the training and testing images with random translation, 
	  		with periodic boundary condition.

	core_size : (int, int)
  	  (image_height, image_width). The height and width of the core image. It can be any size, 
  	  larger or smaller than the cropped image. Default (22, 22) to remain the cropped image size
  	  after cropping the margin of MNIST image. Only effective when mode != "original".

	target_image_size : (int, int)
	  (target_image_height, target_image_width). The target image height and width. If the 
	  core_size is smaller than the target_image_size, this function will pad the image to 
	  the right or bottom. Only effective when mode != "original".

	digit_per_class : int
	  Number of digits per class in the traning set. If the number of digits is smaller or 
	  equal than the available digits in that class, it will sample without replacement, 
	  otherwise will sample with replacement. Default None to use all the given data.

	one_hot : bool
	  whether to return the y_train and y_test as one_hot vector. Default False.
	"""
	np.random.seed(42) # Seed the random number generator
	dataset = fetch_mldata('MNIST original')
	X_flattened = dataset.data
	y = dataset.target.astype(int)
	# Reshaping the examples into the shape of (-1, 28, 28)
	X = np.array([np.reshape(f, (-1, 28)) for f in X_flattened])
	# Always split the MNIST training set and testing set in the traditional way:
	X_train, X_test = X[:60000], X[60000:]
	y_train, y_test = y[:60000], y[60000:]
	# sample training data with each class having digit_per_class of digits
	X_train, y_train = sample_digits(X_train, y_train, digit_per_class = digit_per_class)
	print("MNIST dataset fetched. Length: training set: {0} images, testing set {1} images".format(len(X_train), len(X_test)))
	if mode != "original":
		# Crop and shrink the images:
		X_train = crop_shrink_pad_image(X_train, core_size = core_size, target_image_size = target_image_size)
		X_test = crop_shrink_pad_image(X_test, core_size = core_size, target_image_size = target_image_size)
		# randomly translate the image within the boundary:
		if mode == "random translation":
			X_train = random_translate_image(X_train, row_limit = (0, target_image_size[0] - core_size[0]), 
								   column_limit = (0, target_image_size[1] - core_size[1]))
			X_test = random_translate_image(X_test, row_limit = (0, target_image_size[0] - core_size[0]), 
								   column_limit = (0, target_image_size[1] - core_size[1]))
		# randomly translate the image with periodic boundary condition:
		elif mode == "random translation torus":
			X_train = random_translate_image(X_train, row_limit = (0, target_image_size[0]), 
								   column_limit = (0, target_image_size[1]))
			X_test = random_translate_image(X_test, row_limit = (0, target_image_size[0]), 
								   column_limit = (0, target_image_size[1]))
		else:
			raise Exception("Mode not recognized! Please choose from the available modes!")
	if one_hot:
		from util import dense_to_one_hot
		y_train = dense_to_one_hot(y_train, num_classes = 10)
		y_test = dense_to_one_hot(y_test, num_classes = 10)
	X_train = X_train / float(255)
	X_test = X_test / float(255)
	return (X_train, y_train), (X_test, y_test)


def sample_digits(X, y, digit_per_class = None):
	"""This function sample the data in the X, with each class having digit_per_class
	number of images. y is the labels. For digit_per_class, default None to use all the given data
	"""
	if digit_per_class is None:
		return X, y
	else:
		for digit in range(10):
			X_digit = X[y == digit]
			y_digit = y[y == digit]
			num_digit_available = len(X_digit)
			if digit_per_class <= num_digit_available:
				idx = np.random.choice(num_digit_available, size = digit_per_class, replace = False)
			else:
				idx = np.random.choice(num_digit_available, size = digit_per_class, replace = True)
			X_digit = X_digit[idx]
			y_digit = y_digit[idx]
			# Combine different sampled digits together
			if digit == 0:
				X_combined = X_digit
				y_combined = y_digit
			else:
				X_combined = np.concatenate((X_combined, X_digit))
				y_combined = np.concatenate((y_combined, y_digit))
		return X_combined, y_combined


def crop_shrink_pad_image(X, crop_margin = (3,3), core_size = None, target_image_size = None):
	"""This function first crop the images' margin, then resize the image to the core_size,
	, then pads the image to the right and bottom so that the final image size is target_image_size

	Parameters
	----------
	  X : np.array
		Input feature matrix (N, row, column), 3D numpy array

	  crop_margin : (int, int)
		The row-wise and column-wise margin of the image to crop. Default is (3, 3) for MNIST,
		so that the (28, 28) MNIST image becomes (22, 22)

	  core_size : (int, int)
	  	(image_height, image_width). The height and width of the core image. It can be any size, 
	  	larger or smaller than the cropped image. Default None to remain the cropped image size.

	  target_image_size : (int, int)
		(target_image_height, target_image_width). The target image height and width. If the 
		core_size is smaller than the target_image_size, this function will pad the image to 
		the right or bottom. Default None to remain the core_size.
	"""
	image_shape = np.shape(X)[1:]
	assert crop_margin[0] * 2 < image_shape[0], "The vertical crop_margin exceeds half the image height!"
	assert crop_margin[1] * 2 < image_shape[1], "The horizontal crop_margin exceeds half the image width!"
	# Crop the image:
	X_resized = X[:, crop_margin[0]:image_shape[0]-crop_margin[0], crop_margin[1]:image_shape[1]-crop_margin[1]]
	# Resize the image to core_size:
	if core_size is not None and core_size != np.shape(X_resized)[1:]:
		import scipy
		X_resized = np.array([scipy.misc.imresize(X_resized[i], size = core_size) for i in range(len(X_resized))])
	# Pad the image so that it has the target_image_size:
	if target_image_size is not None and target_image_size != core_size:
		current_size = np.shape(X_resized)[1:]
		assert current_size[0] <= target_image_size[0] and current_size[1] <= target_image_size[1],\
		"The image's core_size should be equal or smaller than its target_image_size!"
		X_resized = np.lib.pad(X_resized, ((0, 0), (0, target_image_size[0] - current_size[0]),
			(0, target_image_size[1] - current_size[1])), 'constant', constant_values=0)

	return X_resized


def random_translate_image(X, row_limit, column_limit):
	"""This function randomly cicular-shifts the image both horizontally and vertically

	Parameters
	----------
	  X : np.array
		Input feature matrix (N, row, column), 3D numpy array

	  row_limit : (int, int)
		The lower and higher bound of the image's circular shift vertically. 
		Positive number means that the image is shifted downward.

	  column_limit : (int, int)
		The lower and higher bound of the image's circular shift horizontally. 
		Positive number means that the image is shifted rightward.
	"""
	len_X = len(X)
	row_circ_shift = np.random.randint(low = row_limit[0], high = row_limit[1] + 1, size = (len_X,1))
	column_circ_shift = np.random.randint(low = column_limit[0], high = column_limit[1] + 1, size = (len_X,1))
	circ_shift = np.hstack((row_circ_shift, column_circ_shift))
	return np.array([np.roll(X[i], circ_shift[i], axis = (0,1)) for i in range(len_X)])
