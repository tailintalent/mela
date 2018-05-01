dataset_PATH = "dataset"
is_om = False
if not is_om:
	pytorch_PATH = "../data/pytorch"
	variational_PATH = "../data/variational"
	variational_model_PATH = "../data/variational"
	current_PATH = ""
else:
	pytorch_PATH = "/om/user/tailin/.etc/util/pytorch"
	variational_PATH = "/om/user/tailin/.etc/util/var"
	variational_model_PATH = "/om/user/tailin/.etc/util/vvar/.exp"
	current_PATH = "/home/tailin/.var/src/AI_scientist/"