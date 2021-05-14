import os,time,cv2, sys, math
import tensorflow as tf

import argparse
import numpy as np
from utils import utils, helpers
from ModelBuilders import ModelBuilders

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default=None, required=True, help='The image you want to predict')
parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The local path for your latest checkpoint')
parser.add_argument('--crop_height', type=int, default=480, help='Height of the cropped image')
parser.add_argument('--crop_width', type=int, default=640, help='Width of the cropped image')
parser.add_argument('--model', type=str, default=None, required=True, help='The actual model you are using for traning/validation and testing purpose')
parser.add_argument('--dataset', type=str, default="MyAISProjectImages", required=False, help='The Dataset direcroy name you are referring to')
args = parser.parse_args()
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_names.csv"))
num_classes = len(label_values)

print("\n*****##Begin prediction *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Number of Classes -->", num_classes)
print("Image -->", args.image)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 
network, _ = ModelBuilders.build_model(args.model, net_input=net_input,
                                        num_classes=num_classes,
                                        crop_width=args.crop_width,
                                        crop_height=args.crop_height,
                                        is_training=False)
sess.run(tf.global_variables_initializer())
print('Loading the PSPNet model checkpoint')
saver=tf.train.Saver(max_to_keep=100)
saver.restore(sess, args.checkpoint_path)
print("Testing the trained/validated image " + args.image)
loaded_image = utils.load_image(args.image)
resized_image =cv2.resize(loaded_image, (args.crop_width, args.crop_height))
input_image = np.expand_dims(np.float32(resized_image[:args.crop_height, :args.crop_width]),axis=0)/255.0
st = time.time()
output_image = sess.run(network,feed_dict={net_input:input_image})
run_time = time.time()-st
output_image = np.array(output_image[0,:,:,:])
output_image = helpers.reverse_one_hot(output_image)
out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
file_name = utils.filepath_to_name(args.image)
cv2.imwrite("%s_PredictedImage.png"%(file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
print("")
print("Image Prediction done!")
print("Saved image " + "%s_pred.png"%(file_name))
#---------------------------------------------------END-----------------------------------------