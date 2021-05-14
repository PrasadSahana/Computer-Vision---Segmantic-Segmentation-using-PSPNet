import sys, os
import tensorflow as tf
import subprocess
from models.PSPNet import build_pspnet
sys.path.append("models")

SUPPORTED_MODEL = ["PSPNet"]

SUPPORTED_FRONTENDS = ["ResNet50", "ResNet101", "ResNet152"]

def download_checkpoints(model_name):
    subprocess.check_output(["python", "utils/Download_Pretrained_Checkpoints.py", "--model=" + model_name])
def build_model(model_name, net_input, num_classes, crop_width, crop_height, frontend="ResNet101", is_training=True):
	print("Fetching and Preparing the model to train ...")
	if model_name not in SUPPORTED_MODEL:
		raise ValueError("The model is not supported. We have written this script only for the following model: {0}".format(SUPPORTED_MODEL))

	if frontend not in SUPPORTED_FRONTENDS:
		raise ValueError("The backbone model is not supported. We have written this script only for the following model:: {0}".format(SUPPORTED_FRONTENDS))

	if "ResNet50" == frontend and not os.path.isfile("models/resnet_v2_50.checkpoint"):
	    download_checkpoints("ResNet50")
	if "ResNet101" == frontend and not os.path.isfile("models/resnet_v2_101.checkpoint"):
	    download_checkpoints("ResNet101")
	if "ResNet152" == frontend and not os.path.isfile("models/resnet_v2_152.checkpoint"):
	    download_checkpoints("ResNet152")
	network = None
	init_fn = None
	if model_name == "PSPNet":
	    network, init_fn = build_pspnet(net_input, label_size=[crop_height, crop_width], preset_model = model_name, frontend=frontend, num_classes=num_classes, is_training=is_training)  #
	else:
	    raise ValueError("Error: the model %d is not supported. Try with the models which are available using the command 'python main.py --help' ")
	return network, init_fn