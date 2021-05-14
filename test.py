import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
from utils import utils, helpers
from ModelBuilders import ModelBuilders

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The local path for your latest checkpoint')
parser.add_argument('--crop_height', type=int, default=640, help='Height of the cropped image')
parser.add_argument('--crop_width', type=int, default=480, help='Width of the cropped image')
parser.add_argument('--model', type=str, default=None, required=True, help='The actual model you are using for traning/validation and testing purpose')
parser.add_argument('--dataset', type=str, default="MyAISProjectImages", required=False, help='The Dataset direcroy name you are referring to')
args = parser.parse_args()
print("Fetching the dataset info. ...")
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_names.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name
num_classes = len(label_values)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 
network, _ = ModelBuilders.build_model(args.model, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=False)
sess.run(tf.global_variables_initializer())
print('Loading PSP Net model checkpoint!!!')
saver=tf.train.Saver(max_to_keep=100)
saver.restore(sess, args.checkpoint_path)
print("Fetching the data ...")
train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=args.dataset)
if not os.path.isdir("%s"%("TestData")):
        os.makedirs("%s"%("TestData"))
target=open("%s/TestData_Score.csv"%("TestData"),'w')
target.write("TestData_Name, TestData_Accuracy, Precision, Recall, F1 Score, MeanValue IoU, %s\n" % (class_names_string))
scores_list = []
class_scores_list = []
precision_list = []
recall_list = []
f1_list = []
iou_list = []
run_times_list = []
for ind in range(len(test_input_names)):
    sys.stdout.write("\rRunning the Test Image %d / %d"%(ind+1, len(test_input_names)))
    sys.stdout.flush()
    input_image = np.expand_dims(np.float32(utils.load_image(test_input_names[ind])[:args.crop_height, :args.crop_width]),axis=0)/255.0
    gt = utils.load_image(test_output_names[ind])[:args.crop_height, :args.crop_width]
    gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))
    st = time.time()
    output_image = sess.run(network,feed_dict={net_input:input_image})
    run_times_list.append(time.time()-st)
    output_image = np.array(output_image[0,:,:,:])
    output_image = helpers.reverse_one_hot(output_image)
    out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
    accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes)
    file_name = utils.filepath_to_name(test_input_names[ind])
    target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))  #
    for item in class_accuracies:
        target.write(", %f"%(item))
    target.write("\n")
    scores_list.append(accuracy)
    class_scores_list.append(class_accuracies)
    precision_list.append(prec)
    recall_list.append(rec)
    f1_list.append(f1)
    iou_list.append(iou)
    gt = helpers.colour_code_segmentation(gt, label_values)
    cv2.imwrite("%s/%s_PredictedImage.png"%("TestDataPredicted", file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
    cv2.imwrite("%s/%s_UserImage.png"%("TestDataImage", file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))
target.close()
avg_score = np.mean(scores_list)
class_avg_scores = np.mean(class_scores_list, axis=0)
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)
avg_f1 = np.mean(f1_list)
avg_iou = np.mean(iou_list)
avg_time = np.mean(run_times_list)
print("Average accuracy of the TestData= ", avg_score)
print("Average accuracy of the TestData(per class) = \n")
for index, item in enumerate(class_avg_scores):
    print("%s = %f" % (class_names_list[index], item))
print("Average Precision = ", avg_precision)
print("Average Recall = ", avg_recall)
print("Average F1 score = ", avg_f1)
print("Average Mean IoU Value = ", avg_iou)
print("Average Run Time = ", avg_time)
#---------------------------------------------------END-----------------------------------------