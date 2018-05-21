# Meta-Learning Autoencoders (MeLA)

This repository implements Meta-Learning Autoencoders (MeLA), a framework that can transform a model that is originally intended for single-task learning into one that can quickly adapt to new tasks with few examples without training. It augments  the single-task model with a meta-recognition model which learns a succinct task representation using just a few informative examples, and a meta-generative model to construct parameters for the task-specific model. MeLA has 3 core capabilities:

1. **Augmented model recognition**: MeLA strategically builds on a pre-existing, single-task trained network, augmenting it with a second network used for meta-recognition. It achieves lower loss in unseen environments at zero and few gradient steps, compared both the original architecture upon which it is based and with state-of-the-art meta-learning algorithms.
2. **Influence identification**: MeLA facilitates interpretability by identifying the which exam- ples are most useful for determining the model (for example, a rectangle’s vertices have far greater influence in determining its position and size than its inferior points), and
3. **Interactive learning**: MeLA can actively request new samples which maximize its ability to learn models.



# Requirements

- Python 3.5.2 (Python 2.7 is also fine, except for running Gym environments)
- [PyTorch 0.3](http://pytorch.org/previous-versions/) (Do not use PyTorch 0.4 since it has several APIs not backward compatible)
- [baselines](https://github.com/openai/baselines) for running Gym environments (needs Python 3.5)
- scikit-learn
- matplotlib

After installing the above requirements, install the custom Gym environment by first going to the gym/ folder, then run:

	$ pip3 install -e .

