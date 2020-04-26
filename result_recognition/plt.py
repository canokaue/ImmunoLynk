import cv2
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import matplotlib  
matplotlib.use('TkAgg') # macos backend
import matplotlib.pyplot as plt
import scipy as sc

img = cv2.imread('trimg/image0.jpg')
# convert to numpy array
res = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
data = res
# expand dimension to one sample

samples = expand_dims(data, 0)
# create image data augmentation generator
datagen = ImageDataGenerator(rotation_range = 90, width_shift_range=[-100,100])
# prepare iterator
it = datagen.flow(samples, batch_size=1)
for i in range(9):
	# define subplot
	plt.subplot(330 + 1 + i)
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
	image = batch[0].astype('uint8')
	# plot raw pixel data
	plt.imshow(image)
# show the figure
plt.show()