python main.py --datasource=tanh --logdir=logs/tanhh1/ --metatrain_iterations=5000 --norm=None --update_batch_size=5

python main.py --datasource=tanh --logdir=logs/tanhh2/ --metatrain_iterations=5000 --norm=None --update_batch_size=10

python main.py --datasource=tanh --logdir=logs/tanhh3/ --metatrain_iterations=5000 --norm=None --update_batch_size=2

python main.py --datasource=tanh --logdir=logs/tanhh3/ --metatrain_iterations=5000 --norm=None --update_batch_size=2 --meta_lr .00001 --update_lr .001
