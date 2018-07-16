# set up Python environment: numpy for numerical routines, and matplotlib for plotting

import numpy as np

import os

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib



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
BATCH = 5

with open('./val_imgs_med2.txt') as f:
     for i, fs in enumerate(f):
         if i < BATCH:
	     fs = fs[:-1]
             print('%d %s...\n') % (i, fs)
             image = caffe.io.load_image(fs)
             transformed_image = transformer.preprocess('data', image)
             net.blobs['data'].data[i] = transformed_image
### perform classification

output = net.forward()


output_prob = output['prob'][0]  # the output probability vector for the first image in the batch


print 'predicted class of first image is:', output_prob.argmax()

print 'predicted class of second image is:', output['prob'][1].argmax()



# load ImageNet labels

labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'

if not os.path.exists(labels_file):

	os.system('../data/ilsvrc12/get_ilsvrc_aux.sh')

    

labels = np.loadtxt(labels_file, str, delimiter='\t')



print 'output label:', labels[output_prob.argmax()]

print 'output label:', labels[output['prob'][1].argmax()]


# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

# We'll generate an animation with matplotlib and moviepy.
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 1001)) #as 1000 classes 1001th for non activation
    print palette
    print palette[1000]
    palette[1000]=[1,1,1]
   
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.

    return f, ax, sc


def vis_square(data, filename):

	"""Take an array of shape (n, height, width) or (n, height, width, 3)

   	and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    

	# normalize data for display

	data = (data - data.min()) / (data.max() - data.min())

    

	# force the number of filters to be square

	n = int(np.ceil(np.sqrt(data.shape[0])))

	padding = (((0, n ** 2 - data.shape[0]),

           	(0, 1), (0, 1))             	# add some space between filters

           	+ ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)

	data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    

	# tile the filters into an image

	data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))

	data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    

	im=plt.imshow(data); plt.axis('off')

	plt.colorbar(im)

	plt.savefig(filename)



def bar_plot(data, x_label, y_label):

  ind = np.arange(len(data))

  width = 0.35 



  # normalize data for display

  data = (data - data.min()) / (data.max() - data.min())



  fig, dx = plt.subplots()

  rect1= dx.bar(ind,data,width,color='b')



  # add some text for labels, title and axes ticks

  dx.set_ylabel('Activations')

  dx.set_title('Nuerons')

  dx.set_xticks(ind + width / 2)

  #dx.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))



  plt.show()



model = TSNE(n_components=2, random_state=0)

print 'shape filter conv1', net.params['conv1'][0].data.shape

print 'shape conv1', net.blobs['conv1'].data.shape

print 'conv1[0,0,0,0]', net.blobs['conv1'].data[0,0,0,0]

print 'conv1[1,0,0,0]', net.blobs['conv1'].data[1,0,0,0]

#print net.blobs['conv1'].data

f11 = net.blobs['conv1'].data[0, 0, :, :]

f11_1d = f11.ravel()

#print f11

#print 'Base'

#print f11_1d

print net.blobs

#batch_activation = [[0 for x in range(3025)] for y in range(BATCH*96)]
#batch_color = np.hstack(i for i in range(96*BATCH))
#print 'batch_color shape \n', batch_color.shape

for i in range(0,BATCH):
    fi = net.blobs['conv1'].data[i, :, :, :]
    fi_avg = np.mean(fi, (1, 2))
   # print 'Avg activation shape', fi_avg.shape
  #  print 'Avg activation', fi_avg
    norm1 = fi_avg/np.linalg.norm(fi_avg)
 #   print 'Normalized Avg activation', norm1
    norm1_asarray=np.asarray(norm1)
    less_20_perc = norm1_asarray < 0.2
    norm1_asarray[less_20_perc] = 0
    print 'Normalized Avg activation after eliminating low activation', norm1_asarray
    #bar_plot(fi_avg, 'Nuerons', 'Activations')
    print norm1_asarray
    
    more_20_perc = norm1_asarray > 0.1
    y = np.hstack(i for i in range(96))
    
    y[less_20_perc]=1000
    y[more_20_perc]=output['prob'][i].argmax()
    #print y
    #color_palette
    reshaped_fi = fi.reshape(96,3025)
    #model = TSNE().fit_transform(reshaped_fi)
    #scatter(model, y)
 #   print reshaped_fi
    if i == 0 :
       batch_activation = reshaped_fi
       batch_color = y
    if i > 0:
       batch_activation = np.concatenate((batch_activation,reshaped_fi))
       batch_color = np.concatenate((batch_color,y))
    if i==BATCH-1:
      # print batch_color
       model = TSNE().fit_transform(batch_activation)
       scatter(model, batch_color)
       plt.savefig('digits_tsne-generated.png')

print batch_activation.shape
#    batch_activation[i*96:]=[row[:] for row in reshaped_fi]
#    batch_color[i*96:]=y
 #   print 'y value \n', y

#print 'batch_color \n ', batch_color
#print 'batch_color shape \n', batch_color.shape
#print 'batch_activation \n', batch_activation
#model = TSNE().fit_transform(batch_activation)
#scatter(model, batch_color)
#scatter(model, y)
#print batch_activation
#    plt.show()

#plt.savefig('digits_tsne-generated.png')



filters = net.params['conv1'][0].data

#vis_square(filters.transpose(0, 2, 3, 1),'filter.pdf')



#feat = net.blobs['conv1'].data[0]

#vis_square(feat,'conv1Activation.pdf')



fi = net.blobs['conv2'].data[0, :, :, :]

fi_avg = np.mean(fi, (1, 2))

print 'Avg activation shape', fi_avg.shape

#bar_plot(fi_avg, 'Nuerons', 'Activations')



print 'shape conv2', net.blobs['conv2'].data.shape

feat = net.blobs['conv2'].data[0]

#vis_square(feat,'conv2Activation.pdf')

