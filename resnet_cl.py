import argparse
import os
import sys
import cv2
import math
import h5py
import warnings
import numpy as np
import keras as keras
import tensorflow as tf
from tqdm import tqdm
import matplotlib.image as image
from scipy.misc import imresize

from keras import layers
from keras import initializers
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
import keras.backend as K
from keras.callbacks import TensorBoard
from keras.models import load_model

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='ResNet')
	parser.add_argument('-s','--scale', dest='scale', help='Training image downscale factor. Choose 1,4,8 (default = 1(ground truth))',
	                    default=1, type=int)
	parser.add_argument('-g','--srgan', dest='srgan', help='Use SRGAN preprocessing? Must have SRGAN model.',
	                    action='store_true')
	parser.add_argument('-ng','--no-srgan', dest='srgan', help='Use SRGAN preprocessing? Must have SRGAN model.',
	                    action='store_false')
	if len(sys.argv) == 1:
	    parser.print_help()
	    sys.exit(1)
	args = parser.parse_args()
	print('Called with args:',args)
	assert args.scale in [1,4,8],'Select Scale from [1,4,8]'

	def loaddata(path):
	    with h5py.File(path, 'r') as f:
	    # f = h5py.File('imagenet10.hdf5', 'r')
	        x = [i[...] for i in f["x"].values()]
	        y = f["y"][...]
	        class_names = f["metadata"][...]
	    print("X data size:", len(x))

	    x_new=np.asarray(x)
	    ###PINARDY: Smaller dataset sample for testing
	    #  idx_range = 9557
	    idx_range = len(x) #Orignal

	    #Randomize 
	    idx = np.random.permutation(len(x))[:idx_range]
	    # idx = np.arange(len(x)) #TESTING
	    x = np.asarray(x)[idx]
	    y = y[idx]
	    y_oh = np.eye(len(class_names))[y]
	    return x, y, y_oh

	#Load selected dataset
	x_train, y_train, y_train_oh = loaddata('dataset/imagenet10_train.hdf5')
	x_valid, y_valid, y_valid_oh = loaddata('dataset/imagenet10_valid.hdf5')
	x_test, y_test, y_test_oh = loaddata('dataset/imagenet10_test.hdf5')

	#Classifier / SRGAN
	def img_crop_downsample(img, downscale):
	    ht, wt = 224,224
	    in_ht, in_wt = img.shape[0:2]
	    if in_ht < in_wt:
	        img = imresize(img, (ht,int(in_wt*ht/in_ht)))
	        in_ht, in_wt = img.shape[0:2]
	    else:
	        img = imresize(img, (int(in_ht*wt/in_wt),wt))
	        in_ht, in_wt = img.shape[0:2]

	    dx_ht = in_ht - ht
	    dx_wt = in_wt - wt
	    
	    low_ht = np.random.randint(0, dx_ht + 1)
	    low_wt = np.random.randint(0, dx_wt + 1)

	    hr = img[low_ht:low_ht+ht, low_wt:low_wt+ht, :]
	    lr = imresize(hr, 1/downscale)
	    
	    if classifier_downup:
	        hr = imresize(lr, (ht,wt))
	        
	    #Normalize
	    hr = ((hr - hr.min())/(hr.max() - hr.min()) * 2) - 1
	    lr = (lr - lr.min())/(lr.max() - lr.min())
	    return hr,lr

	def write_log(callback, names, logs, batch_no):
	    for name, value in zip(names, logs):
	        summary = tf.Summary()
	        summary_value = summary.value.add()
	        summary_value.simple_value = value
	        summary_value.tag = name
	        callback.writer.add_summary(summary, batch_no)
	        callback.writer.flush()

	### CHANGE HYPERPARAMETERS HERE
	downsample = args.scale
	epoch = 100
	mb_size = 32
	period = x_train.shape[0]//mb_size
	use_srgan = args.srgan #If training classifier with srgan.
	if use_srgan or downsample == 1:
	    classifier_downup = False #FALSE if training HR or with GAN preprocessing. (224 size) 
	else:
	    classifier_downup = True #TRUE if training LR with no gan. (224//downsample size)
    
	retrain = True #For model finetuning. Else specify none
	update_lr = 1e-5 #For model finetuning new learning rate.
	if retrain:
		if downsample == 1:
			model_dir = 'Resnet_GT/'
			loadmodel_dir = model_dir+'epoch100/Adam99.h5'
		elif use_srgan:
			model_dir = 'Resnet_SRGAN%d/'%downsample
			loadmodel_dir = model_dir+'epoch100/Adam_srgan%d_99.h5'%downsample
		else:
			model_dir = 'Resnet_down%d/'%downsample
			loadmodel_dir = model_dir+'epoch100/Adam_down%d_99.h5'%downsample

	hr,lr=img_crop_downsample(x_train[1], downsample)
	hr = (hr-hr.min())/(hr.max()-hr.min())
	lr = (lr-lr.min())/(lr.max()-lr.min())
	print("HIGH RESOLUTION for use without SRGAN", hr.shape)
	print("LOW RESOLUTION for use SRGAN", lr.shape)

	#import SRGAN model for generator model training
	if use_srgan:
	    config = tf.ConfigProto()
	    config.gpu_options.allow_growth = True
	    sess_g = tf.Session(config = config)
	    saver = tf.train.import_meta_graph('./SRGAN_%d/msevgg/trained_model/srgan_4.model-0.meta'%downsample)
	    saver.restore(sess_g,tf.train.latest_checkpoint('./SRGAN_%d/msevgg/trained_model/'%downsample))
	    graph = tf.get_default_graph()
	    Z = graph.get_tensor_by_name("Z:0")
	    G = graph.get_tensor_by_name("generator:0")
	    tf.reset_default_graph()
	    
	    gen_hr = sess_g.run(G, feed_dict={Z:np.expand_dims(lr,axis=0)})[0]
	    gen_hr = (gen_hr-gen_hr.min())/(gen_hr.max()-gen_hr.min())
	    print("Generated SRGAN upsample", gen_hr.shape)

	def identity_block(input_tensor, kernel_size, filters, stage, block):
	    filters1, filters2, filters3 = filters
	    if K.image_data_format() == 'channels_last':
	        bn_axis = 3
	    else:
	        bn_axis = 1
	    conv_name_base = 'res' + str(stage) + block + '_branch'
	    bn_name_base = 'bn' + str(stage) + block + '_branch'

	    a = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
	    a = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(a)
	    a = Activation('relu')(a)

	    a = Conv2D(filters2, kernel_size,
	               padding='same', name=conv_name_base + '2b')(a)
	    a = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(a)
	    a = Activation('relu')(a)

	    a = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(a)
	    a = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(a)

	    a = layers.add([a, input_tensor])
	    a = Activation('relu')(a)
	    return a

	def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

	    filters1, filters2, filters3 = filters
	    if K.image_data_format() == 'channels_last':
	        bn_axis = 3
	    else:
	        bn_axis = 1
	    conv_name_base = 'res' + str(stage) + block + '_branch'
	    bn_name_base = 'bn' + str(stage) + block + '_branch'

	    a = Conv2D(filters1, (1, 1), strides=strides,
	               name=conv_name_base + '2a')(input_tensor)
	    a = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(a)
	    a = Activation('relu')(a)

	    a = Conv2D(filters2, kernel_size, padding='same',
	               name=conv_name_base + '2b')(a)
	    a = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(a)
	    a = Activation('relu')(a)

	    a = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(a)
	    a = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(a)

	    shortcut = Conv2D(filters3, (1, 1), strides=strides,
	                      name=conv_name_base + '1')(input_tensor)
	    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

	    a = layers.add([a, shortcut])
	    a = Activation('relu')(a)
	    return a

	def ResNet50(include_top=True,input_tensor=None,input_shape=None,pooling=None,classes=1000):
	    # Determine proper input shape
	    input_shape = _obtain_input_shape(input_shape,
	                                      default_size=224,
	                                      min_size=197,
	                                      data_format=K.image_data_format(),
	                                      require_flatten=include_top)

	    if input_tensor is None:
	        img_input = Input(shape=input_shape)
	    else:
	        if not K.is_keras_tensor(input_tensor):
	            img_input = Input(tensor=input_tensor, shape=input_shape)
	        else:
	            img_input = input_tensor
	    if K.image_data_format() == 'channels_last':
	        bn_axis = 3
	    else:
	        bn_axis = 1

	    a = ZeroPadding2D((3, 3))(img_input)
	    a = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(a)
	    a = BatchNormalization(axis=bn_axis, name='bn_conv1')(a)
	    a = Activation('relu')(a)
	    a = MaxPooling2D((3, 3), strides=(2, 2))(a)

	    a = conv_block(a, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
	    a = identity_block(a, 3, [64, 64, 256], stage=2, block='b')
	    a = identity_block(a, 3, [64, 64, 256], stage=2, block='c')

	    a = conv_block(a, 3, [128, 128, 512], stage=3, block='a')
	    a = identity_block(a, 3, [128, 128, 512], stage=3, block='b')
	    a = identity_block(a, 3, [128, 128, 512], stage=3, block='c')
	    a = identity_block(a, 3, [128, 128, 512], stage=3, block='d')

	    a = conv_block(a, 3, [256, 256, 1024], stage=4, block='a')
	    a = identity_block(a, 3, [256, 256, 1024], stage=4, block='b')
	    a = identity_block(a, 3, [256, 256, 1024], stage=4, block='c')
	#     a = identity_block(a, 3, [256, 256, 1024], stage=4, block='d')
	#     a = identity_block(a, 3, [256, 256, 1024], stage=4, block='e')
	#     a = identity_block(a, 3, [256, 256, 1024], stage=4, block='f')

	#     a = conv_block(a, 3, [512, 512, 2048], stage=5, block='a')
	#     a = identity_block(a, 3, [512, 512, 2048], stage=5, block='b')
	#     a = identity_block(a, 3, [512, 512, 2048], stage=5, block='c')

	    a = AveragePooling2D((7, 7), name='avg_pool')(a)

	    if include_top:
	        a = Flatten()(a)
	        a = Dense(classes, activation='softmax', name='fc1000')(a)
	    else:
	        if pooling == 'avg':
	            a = GlobalAveragePooling2D()(a)
	        elif pooling == 'max':
	            a = GlobalMaxPooling2D()(a)

	    # Ensure that the model takes into account
	    # any potential predecessors of `input_tensor`.
	    if input_tensor is not None:
	        inputs = get_source_inputs(input_tensor)
	    else:
	        inputs = img_input
	    # Create model.
	    model = Model(inputs, a, name='resnet50')

	    return model

	#If retraining/ model finetuning
	if loadmodel_dir:
	    from keras.models import load_model
	    model = load_model(loadmodel_dir)
	    if update_lr:
	        print("Old lr:", K.eval(model.optimizer.lr))
	        K.set_value(model.optimizer.lr, update_lr)
	        print("New lr:", K.eval(model.optimizer.lr))
	else:
	    model = ResNet50(include_top=True, classes=10)
	    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4),loss='categorical_crossentropy',metrics=['accuracy'])
	    # model.compile(optimizer=keras.optimizers.SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True),loss='categorical_crossentropy',metrics=['accuracy'])

	# SRGAN
	log_path = './'+model_dir+'epoch200/'
	callback = TensorBoard(log_path)
	callback.set_model(model)
	train_names = ['train_loss', 'train_mae']
	val_names = ['val_loss', 'val_mae']
	validation_size = x_valid.shape[0]
	val_steps = int(math.ceil(validation_size/mb_size))
	#for e in range(epoch):
	for e in range(epoch):
	    print("Epoch"+str(e))
	    perm = np.random.permutation(x_train.shape[0])
	        
	    for i in tqdm(range(period)):
	        train_batch = x_train[perm[i*mb_size:(i+1)*mb_size]]
	        train_label = y_train_oh[perm[i*mb_size:(i+1)*mb_size]]
	        train_batch_hr, train_batch_lr = list(zip(*map(img_crop_downsample, train_batch, np.ones(mb_size)*downsample)))
	        
	        #Train_batch_hr refers to original img dimension
	        if use_srgan:
	            train_batch_lr = np.asarray(train_batch_lr)
	            train_batch_hr = sess_g.run(G, feed_dict={Z:train_batch_lr})
	        else:
	            train_batch_hr=np.asarray(train_batch_hr)
	            
	        #Flip left right to augment
	        if e%2 == 0:
	            train_batch_hr = np.flip(train_batch_hr,axis=2)
	        
	        #train on a batch
	        logs=model.train_on_batch(train_batch_hr,train_label)
	        
	    #Training loss/mean accuracy log
	    write_log(callback, train_names, logs, e)
	    
	    #Validation loss/mean accuracy log
	    logs = np.asarray([0,0])
	    for j in range(val_steps):
	        start = j*mb_size
	        if (j+1)*mb_size > validation_size:
	            end = validation_size
	        else:
	            end = (j+1)*mb_size
	        
	        valid_batch_hr,valid_batch_lr = list(zip(*map(img_crop_downsample, x_valid[start:end], np.ones(end-start)*downsample)))
	        if use_srgan:
	            valid_batch_lr = np.asarray(valid_batch_lr)
	            valid_batch_hr = sess_g.run(G, feed_dict={Z:valid_batch_lr})
	        else:
	            valid_batch_hr = np.asarray(valid_batch_hr)
	            
	        logs= logs + np.asarray(model.test_on_batch(valid_batch_hr,y_valid_oh[start:end]))*(end-start)
	    logs = logs/validation_size
	    write_log(callback, val_names, logs, e)
	    
	    #Save model
	    if downsample == 1:
	        model.save(model_dir+'epoch200/Adam%d.h5'%(e))
	    elif use_srgan:
	        model.save(model_dir+'epoch200/Adam_srgan%d_%d.h5'%(downsample,e))
	    else:
	        model.save(model_dir+'epoch200/Adam_down%d_%d.h5'%(downsample,e))

	#Compare gen validation images with HR validation models with SRGAN preprocessing 
	if use_srgan:
	    val_loss = np.asarray([0,0])
	    for j in range(val_steps):
	        start = j*mb_size
	        if (j+1)*mb_size > validation_size:
	            end = validation_size
	        else:
	            end = (j+1)*mb_size

	        valid_batch_hr,valid_batch_lr = list(zip(*map(img_crop_downsample, x_valid[start:end], np.ones(end-start)*downsample)))
	        if use_srgan:
	            valid_batch_lr = np.asarray(valid_batch_lr)
	            valid_batch_hr = sess_g.run(G, feed_dict={Z:valid_batch_lr})
	        else:
	            valid_batch_hr = np.asarray(valid_batch_hr)
	        val_loss = val_loss + np.asarray(model.test_on_batch(valid_batch_hr,y_valid_oh[start:end]))*(end-start)
	    val_loss = val_loss/validation_size
	    print("SRGAN Preprocessing %dx upscaling, ResNet"%downsample)
	    print("Generated %dx validation set accuracy:"%downsample,val_loss)

	valid_orig_hr,_ = list(zip(*map(img_crop_downsample, x_valid, np.ones(validation_size)*downsample)))
	valid_orig_hr = np.asarray(valid_orig_hr)
	test_orig = model.test_on_batch(valid_orig_hr,y_valid_oh)

	if classifier_downup:
	    print("Downsample %dx validation set accuracy:"%downsample,test_orig)
	else:
	    print("Original HR validation set accuracy:",test_orig)
