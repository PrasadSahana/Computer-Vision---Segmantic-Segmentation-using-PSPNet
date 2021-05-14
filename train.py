#This code is to train the datasets first and then validate it accordingly. We have input around 71 images for both training and validation
from __future__ import print_function
from utils import utils, helpers
from ModelBuilders import ModelBuilders
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import os,time,cv2, sys, math
import os, sys
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#---------------------------------------------
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value is expected here')
#The below areguments are parameters set for our image training---------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=50, help='You can set number of epochs according to your input iamges ')
parser.add_argument('--epoch_start_i', type=int, default=0, help='You can count number of epochs from here')
parser.add_argument('--checkpoint_step', type=int, default=5, help='After how many steps you want your checkpoint to be saved')
parser.add_argument('--validation_step', type=int, default=1, help='On what epoch you want to perfoma validation process')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on(Only valid while running "predict.py"')
parser.add_argument('--continue_training', type=str2bool, default=False, help='You can continue training from the best saved Checkpoint from your local directory')
parser.add_argument('--dataset', type=str, default="MyAISProjectImages", help='The Dataset direcroy name you are referring to.')
parser.add_argument('--crop_height', type=int, default=480, help='Height of the cropped image')
parser.add_argument('--crop_width', type=int, default=640, help='Width of the cropped image')
parser.add_argument('--batch_size', type=int, default=1, help='Number of images present in every batch')
parser.add_argument('--num_val_images', type=int, default=15, help='The number of images you are using for validation purpose')
parser.add_argument('--h_flip', type=str2bool, default=False, help='Horizontal flip of the image for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=False, help='Vertical flip of the image for data augmentation')
parser.add_argument('--brightness', type=float, default=None, help='Change the image brightness for data augmentation. Maximum bightness change factor is between 0.0 and 1.0 with 0.1 being maximum brightness change of 10%% (+-)')
parser.add_argument('--rotation', type=float, default=None, help='Rotate the image for data augmentation')
parser.add_argument('--model', type=str, default="PSPNet", help='The actual model you are using for traning/validation and testing purpose')
parser.add_argument('--frontend', type=str, default="ResNet101", help='The backbone model you are using for traning/validation and testing purpose')
args = parser.parse_args()
# Data augmentation process on the images------------------------------------------------------------------------------------------------------------
def data_augmentation(input_image, output_image):
    input_image, output_image = utils.random_crop(input_image, output_image, args.crop_height, args.crop_width)
    if args.h_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    if args.v_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 0)
        output_image = cv2.flip(output_image, 0)
    if args.brightness:
        factor = 1.0 + random.uniform(-1.0*args.brightness, args.brightness)
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)  #
        input_image = cv2.LUT(input_image, table)
    if args.rotation:
        angle = random.uniform(-1*args.rotation, args.rotation)
    if args.rotation:
        M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)
    return input_image, output_image
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
network, init_fn = ModelBuilders.build_model(model_name=args.model, frontend=args.frontend, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=True)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output))
opt = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.995).minimize(loss, var_list=[var for var in tf.trainable_variables()])
saver=tf.train.Saver(max_to_keep=100)
sess.run(tf.global_variables_initializer())
utils.count_params()
if init_fn is not None:
    init_fn(sess)
model_checkpoint_name = "checkpoints/latest_model_" + args.model + "_" + args.dataset + ".checkpoint"
if args.continue_training:
    print('Loaded the latest available checkpoint to continue model training')
    saver.restore(sess, model_checkpoint_name)
print("Loading the latest fetched data!!!!!! ...")
train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=args.dataset)
#Training begins from this process .. check your terminal to monitor--------------------------------------------------------------------------------------------
print("\n*****##Begin the training process*****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Number of Epochs -->", args.num_epochs)
print("Batch Size -->", args.batch_size)
print("Number of Classes -->", num_classes)
print("Data Augmentation:")
print("\tVertical Flip -->", args.v_flip)
print("\tHorizontal Flip -->", args.h_flip)
print("\tBrightness Alteration -->", args.brightness)
print("\tRotation -->", args.rotation)
print("")
avg_loss_per_epoch = []
avg_scores_per_epoch = []
avg_iou_per_epoch = []
val_indices = []
num_vals = min(args.num_val_images, len(val_input_names))
random.seed(16)
val_indices=random.sample(range(0,len(val_input_names)),num_vals)
for epoch in range(args.epoch_start_i, args.num_epochs):
    current_losses = []
    cnt=0
    id_list = np.random.permutation(len(train_input_names))
    num_iters = int(np.floor(len(id_list) / args.batch_size))
    st = time.time()
    epoch_st=time.time()
    for i in range(num_iters):
        input_image_batch = []
        output_image_batch = []
        for j in range(args.batch_size):
            index = i*args.batch_size + j
            id = id_list[index]
            input_image = utils.load_image(train_input_names[id])
            output_image = utils.load_image(train_output_names[id])
            with tf.device('/cpu:0'):
                input_image, output_image = data_augmentation(input_image, output_image)
                input_image = np.float32(input_image) / 255.0
                output_image = np.float32(helpers.one_hot_it(label=output_image, label_values=label_values))
                input_image_batch.append(np.expand_dims(input_image, axis=0))
                output_image_batch.append(np.expand_dims(output_image, axis=0))
        if args.batch_size == 1:
            input_image_batch = input_image_batch[0]
            output_image_batch = output_image_batch[0]
        else:
            input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
            output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))
#---------------------------------------------------------------------------------------
 #This is the step where training begins after the images is chosen---------------------
        _,current=sess.run([opt,loss],feed_dict={net_input:input_image_batch,net_output:output_image_batch})
        current_losses.append(current)
        cnt = cnt + args.batch_size
        if cnt % 20 == 0:
            string_print = "You are at : Epoch = %d Count = %d LossOfImageValue = %.4f Time = %.2f"%(epoch,cnt,current,time.time()-st)
            utils.LOG(string_print)
            st = time.time()
    mean_loss = np.mean(current_losses)
    avg_loss_per_epoch.append(mean_loss)
# Create local directory to save checkpoints---------------------------------------------------------------------------------------------------
    if not os.path.isdir("%s/%04d"%("Checkpoints",epoch)):
        os.makedirs("%s/%04d"%("Checkpoints",epoch))
#Local directory to which latest checkpoints are being saved------------------------------------------------------------------------------------ 
    print("Saving the latest available Checkpoint to your local directory")   #
    saver.save(sess,model_checkpoint_name)
    if val_indices != 0 and epoch % args.checkpoint_step == 0:
        print("Saving checkpoint for this epoch")
        saver.save(sess,"%s/%04d/model.checkpoint"%("Checkpoints",epoch))
    if epoch % args.validation_step == 0:
        print("Validation is in process")
        target=open("%s/%04d/ValidationScore.csv"%("Checkpoints",epoch),'w')
        target.write("ValidationName, AverageAccuracy, Precision, Recall, F1 SCore, Mean IoU, %s\n" % (class_names_string))
        scores_list = []
        class_scores_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        iou_list = []
#----------------------------------------------------------------------------------------------
#Doing validation for images (Make sure to keep only small set of images here)----------------
        for ind in val_indices:
            input_image = np.expand_dims(np.float32(utils.load_image(val_input_names[ind])[:args.crop_height, :args.crop_width]),axis=0)/255.0
            gt = utils.load_image(val_output_names[ind])[:args.crop_height, :args.crop_width]
            gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))
            output_image = sess.run(network,feed_dict={net_input:input_image})
            output_image = np.array(output_image[0,:,:,:])
            output_image = helpers.reverse_one_hot(output_image)
            out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
            accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes)   #
            file_name = utils.filepath_to_name(val_input_names[ind])
            target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))   #
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
            file_name = os.path.basename(val_input_names[ind])
            file_name = os.path.splitext(file_name)[0]
            cv2.imwrite("%s/%04d/%s_predictedImage.png"%("Checkpoints",epoch, file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
            cv2.imwrite("%s/%04d/%s_UserImage.png"%("Checkpoints",epoch, file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))
        target.close()
        avg_score = np.mean(scores_list)
        class_avg_scores = np.mean(class_scores_list, axis=0)
        avg_scores_per_epoch.append(avg_score)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_iou = np.mean(iou_list)
        avg_iou_per_epoch.append(avg_iou)
        print("\nAverage validated accuracy for epoch #%04d = %f"% (epoch, avg_score))
        print("Average class(per class) validated accuracy for epoch #%04d:"% (epoch))
        for index, item in enumerate(class_avg_scores):
            print("%s = %f" % (class_names_list[index], item))
        print("Obtained Validation precision = ", avg_precision)
        print("Obtained Validation recall = ", avg_recall)
        print("Obtained Value for F1 score = ", avg_f1)
        print("Obtained Validation Value for IoU = ", avg_iou)
    epoch_time=time.time()-epoch_st
    remain_time=epoch_time*(args.num_epochs-1-epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    if s!=0:
        train_time="Be patient! You have remaining train time of = %d hours %d minutes %d seconds\n"%(h,m,s)
    else:
        train_time="Remaining train time : Training completed Successfully.\n"
    utils.LOG(train_time)
    scores_list = []
#----------------------------------------------------------------------------------------------------
#Monitoring the image learning process through graph, for every saved checkpoint.  
    fig1, ax1 = plt.subplots(figsize=(12, 9))
    ax1.plot(range(epoch+1), avg_scores_per_epoch)
    ax1.set_title("Avg Val Accuracy_VS_Epochs")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Avg Val Accuracy")
    plt.savefig('Accuracy_VS_Epochs.png')
    plt.clf()

    fig2, ax2 = plt.subplots(figsize=(12, 9))
    ax2.plot(range(epoch+1), avg_loss_per_epoch)
    ax2.set_title("Avgloss_VS_Epochs")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    plt.savefig('Loss_VS_Epochs.png')
    plt.clf()

    fig3, ax3 = plt.subplots(figsize=(12, 9))
    ax3.plot(range(epoch+1), avg_iou_per_epoch)
    ax3.set_title("AvgIoU_VS_Epochs")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("IoU")
    plt.savefig('IOU_VS_Epochs.png')
 #-------------------------END--------------------------------------------


