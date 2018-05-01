# AI_scientist

Use meta-learning autoencoder to learn to generate a parametrized class of models, and predict the parameters at the same time. In this way, for new unseen tasks, it

1. Learns faster than learning from scratch.
2. Learns with fewer examples.
3. Discover parameters that determine the important variations of model class.
4. Able to identify informative examples that determine the parameters, and enable machine teaching.
5. Identify the next most important points to measure, when getting new examples or measurements are costly.


# Requirements

- Python 3.5.2 (Python 2.7 is also fine, except for running Gym environments)
- [PyTorch 0.3](http://pytorch.org/previous-versions/) (Do not use PyTorch 0.4 since it has several APIs not backward compatible)
- [baselines](https://github.com/openai/baselines) for running Gym environments (needs Python 3.5)
- scikit-learn
- matplotlib

After installing the above requirements, install the custom Gym environment by first going to the gym/ folder, then run:

	$ pip3 install -e .

to install the developer version of Gym.

