# set up Python environment: numpy for numerical routines, and matplotlib for plotting

import numpy as np

import os

import matplotlib.pyplot as plt

from os import listdir

from os.path import isfile, join

import networkx as nx

import time

from sklearn.manifold import TSNE



# display plots in this notebook

#%matplotlib inline



# set display defaults

plt.rcParams['figure.figsize'] = (10, 10)    	# large images

plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels

plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap



# The caffe module needs to be on the Python path;

#  we'll add it here explicitly.

import sys

caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)

sys.path.insert(0, caffe_root + 'python')



import caffe

# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.



import os

if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):

	print 'CaffeNet found.'

else:

	print 'Downloading pre-trained CaffeNet model...'

	#!../scripts/download_model_binary.py ../models/bvlc_reference_caffenet









caffe.set_mode_cpu()



model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'

model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'



net = caffe.Net(model_def,  	# defines the structure of the model

            	model_weights,  # contains the trained weights

            	caffe.TEST) 	# use test mode (e.g., don't perform dropout)









# load the mean ImageNet image (as distributed with Caffe) for subtraction

mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')

mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

print 'mean-subtracted values:', zip('BGR', mu)



# create transformer for the input called 'data'

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})



transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension

transformer.set_mean('data', mu)        	# subtract the dataset-mean value in each channel

transformer.set_raw_scale('data', 255)  	# rescale from [0, 1] to [0, 255]

transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR



# set the size of the input (we can skip this if we're happy

#  with the default; we can also change it later, e.g., for different batch sizes)

net.blobs['data'].reshape(50,    	# batch size

                      	3,     	# 3-channel (BGR) images

                      	227, 227)  # image size is 227x227



#image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')

#transformed_image = transformer.preprocess('data', image)



# copy the image data into the memory allocated for the net

#net.blobs['data'].data[...] = transformed_image



## load validation images

#mypath = caffe_root + 'python/img_val' 

#val_img = [f for f in listdir(mypath) if isfile(join(mypath, f))]



#for i, f in enumerate(val_img):

#  if i < 50:

    #print("%s \n") %(caffe_root + 'python/img_val/' + f)

#    image = caffe.io.load_image(caffe_root + 'python/img_val/' + f)

#    transformed_image = transformer.preprocess('data', image)

#    net.blobs['data'].data[i] = transformed_image


with open('./val_imgs_med2.txt') as f:
     for i, fs in enumerate(f):
         if i < 50:
             fs = fs[:-1]
             print('%d %s...\n') % (i, fs)
             image = caffe.io.load_image(fs)
             transformed_image = transformer.preprocess('data', image)
             net.blobs['data'].data[i] = transformed_image

zero=np.full((11,11),0)

for i in range(40):
   net.params['conv1'][0].data[i,0,:]=zero
   net.params['conv1'][0].data[i,1,:]=zero
   net.params['conv1'][0].data[i,2,:]=zero
   print net.params['conv1'][0].data[i,0,:]

print net.params['conv1'][0].data[:]



### perform classification
start_time = time.time()
output = net.forward()
elapsed_time = time.time() - start_time

print elapsed_time


filters = net.params['conv1'][0].data

#print filters
start_time = time.time()
output = net.forward()
elapsed_time = time.time() - start_time

print elapsed_time


#print filters


start_time = time.time()
output = net.forward()
elapsed_time = time.time() - start_time

print elapsed_time

start_time = time.time()
output = net.forward()
elapsed_time = time.time() - start_time

print elapsed_time

start_time = time.time()
output = net.forward()
elapsed_time = time.time() - start_time

print elapsed_time

print 'output shape', output['prob'].shape



output_prob = output['prob'][0]  # the output probability vector for the first image in the batch



