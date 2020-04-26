
import collections
from datetime import datetime
from math import ceil, floor
import matplotlib  
matplotlib.use('TkAgg') # macos backend
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import cv2
import tensorflow as tf
import keras
from keras_applications.resnet import ResNet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import ShuffleSplit
from keras import backend as K
from tensorflow.keras.applications import imagenet_utils

preprocess_input = imagenet_utils.preprocess_input

TRAIN_IMAGES_DIR = TEST_IMAGES_DIR = 'downloads/'

WEIGHTS_PATH = 'resnet50_weights.h5'

class TestResModel:
    def __init__(self, engine, input_dims, batch_size=5, num_epochs=4,
                 n_classes=4, learning_rate=1e-3, n_augment = 9,
                 decay_rate=1.0, decay_steps=1, weights=WEIGHTS_PATH, verbose=1):
        self.engine = engine
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.n_augment = n_augment
        self.weights = weights
        self.verbose = verbose
        self._build()
    def _build(self):
        self.engine.trainable = True
        engine = self.engine(include_top=False,
                             weights=self.weights, input_shape=(*self.input_dims[:2], 3),
                             backend = keras.backend, layers = keras.layers, 
                             models = keras.models, utils = keras.utils,)
        set_trainable = False
        for layer in engine.layers:
        #    if layer.name in ['res5c_branch2b', 'res5c_branch2c', 'activation_97']:
        #      set_trainable = True
        #    if set_trainable:
        #      layer.trainable = False
        #    else:
            layer.trainable = False
        x = keras.layers.GlobalAveragePooling2D(name='max_pool')(engine.output)
        out = keras.layers.Dense(self.n_classes, activation="sigmoid", name='dense_output')(x)
        self.model = keras.models.Model(inputs=engine.input, outputs=out)
        # loss function has been changed needs to be investigated.
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=keras.optimizers.Adam(),
                           metrics=['accuracy'])
    def fit_and_predict(self, train_df, valid_df, test_df):
        # callbacks
        pred_history = PredictionCheckpoint(test_df, valid_df,
                                            n_classes=self.n_classes,
                                            batch_size=self.batch_size,
                                            input_size=self.input_dims)
        #checkpointer = keras.callbacks.ModelCheckpoint(filepath='%s-{epoch:02d}.hdf5' % self.engine.__name__, verbose=1, save_weights_only=True, save_best_only=False)
        scheduler = keras.callbacks.LearningRateScheduler(lambda epoch: self.learning_rate * pow(self.decay_rate, floor(epoch / self.decay_steps)))
        self.model.fit_generator(
            DataGenerator(
                list_IDs = train_df.index, 
                img_labels = train_df,
                batch_size=self.batch_size,
                img_size=self.input_dims,
                img_dir=TRAIN_IMAGES_DIR,
                n_classes = self.n_classes,
                train=True,
                n_augment = self.n_augment,
                shuffle = True
            ),
            epochs=self.num_epochs,
            verbose=self.verbose,
            #use_multiprocessing=True,
            #workers=4#,
            #callbacks=[history]
            #callbacks=[tensorboard_callback]
        )
        return pred_history
    def predict(self, image_name, path2image=TRAIN_IMAGES_DIR):
        #### Predict one image at a time
        X = _read( path2image + image_name,self.input_dims,0, plot=False)
        res = self.model.predict(X, batch_size=1)
        return res
    def save(self, path):
        self.model.save_weights(path)
    def load(self, path):
        self.model.load_weights(path)


class MyDeepModel:
    
    def __init__(self, engine, input_dims, batch_size=5, num_epochs=4,
                 n_classes=4, learning_rate=1e-3, n_augment = 9,
                 decay_rate=1.0, decay_steps=1, weights=WEIGHTS_PATH, verbose=1):
        
        self.engine = engine
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.n_augment = n_augment
        self.weights = weights
        self.verbose = verbose
        self._build()

    def _build(self):
        
    
        engine = self.engine(include_top=False,
                             weights=self.weights, input_shape=(*self.input_dims[:2], 3),
                             backend = keras.backend, layers = keras.layers, 
                             models = keras.models, utils = keras.utils,)
        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(engine.output)
        out = keras.layers.Dense(self.n_classes, activation="sigmoid", name='dense_output')(x)

        self.model = keras.models.Model(inputs=engine.input, outputs=out)
        # loss function has been changed needs to be investigated.
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=keras.optimizers.Adam(0.0),
                           metrics=['accuracy'])
    

    def fit_and_predict(self, train_df, valid_df, test_df):
        
        # callbacks
        pred_history = PredictionCheckpoint(test_df, valid_df,
                                            n_classes=self.n_classes,
                                            batch_size=self.batch_size,
                                            input_size=self.input_dims)
        #checkpointer = keras.callbacks.ModelCheckpoint(filepath='%s-{epoch:02d}.hdf5' % self.engine.__name__, verbose=1, save_weights_only=True, save_best_only=False)
        scheduler = keras.callbacks.LearningRateScheduler(lambda epoch: self.learning_rate * pow(self.decay_rate, floor(epoch / self.decay_steps)))
        self.model.fit_generator(
            DataGenerator(
                list_IDs = train_df.index, 
                img_labels = train_df,
                batch_size=self.batch_size,
                img_size=self.input_dims,
                img_dir=TRAIN_IMAGES_DIR,
                n_classes = self.n_classes,
                train=True,
                n_augment = self.n_augment,
                shuffle = True
            ),
            epochs=self.num_epochs,
            verbose=self.verbose,
            use_multiprocessing=True,
            workers=4#,
            #callbacks=[history]
            #callbacks=[tensorboard_callback]
        )
        
        return pred_history

    def predict(self, image_name, path2image=TRAIN_IMAGES_DIR):
        #### Predict one image at a time
        X = _read( path2image + image_name, self.input_dims,0, plot=False)
        res = self.model.predict(X, batch_size=1)
        return res
    
    def save(self, path):
        self.model.save_weights(path)
    
    def load(self, path):
        self.model.load_weights(path)


class PredictionCheckpoint(keras.callbacks.Callback):
    
    def __init__(self, test_df, valid_df, n_classes =4,
                 test_images_dir=TEST_IMAGES_DIR, 
                 valid_images_dir=TRAIN_IMAGES_DIR, 
                 batch_size=32, input_size=(224, 224, 3)):
        
        self.test_df = test_df
        self.valid_df = valid_df
        self.test_images_dir = test_images_dir
        self.valid_images_dir = valid_images_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.n_classes = n_classes
        
        
    def on_train_begin(self, logs={}):
        self.test_predictions = []
        self.valid_predictions = []
        
    def on_epoch_end(self,batch, logs={}):
        self.test_predictions.append(
            self.model.predict_generator(
                DataGenerator(self.test_df.index, test_df,
                              batch_size=self.batch_size, img_size=self.input_size, 
                              img_dir=self.test_images_dir, n_classes = self.n_classes,
                              train =False, n_augment = 0, shuffle=True),
                verbose=2)[:len(self.test_df)])
        
        self.valid_predictions.append(
            self.model.predict_generator(
                DataGenerator(self.valid_df.index, valid_df,
                              batch_size=self.batch_size, img_size=self.input_size, 
                              img_dir=self.valid_images_dir, n_classes = self.n_classes,
                              train =False, n_augment = 0, shuffle=True),
                verbose=2)[:len(self.valid_df)])
        valid_labels = np.zeros((self.valid_df.shape[0], self.n_classes))
        valid_labels[np.arange(self.valid_df.shape[0]), self.valid_df['label']] = 1
        print('valid_labels', valid_labels )
        print('pred_labels', self.valid_predictions)
        print("validation loss: %.4f" %
              weighted_log_loss_metric(valid_labels, 
                                   np.average(self.valid_predictions, axis=0)))


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, img_labels, batch_size=1, img_size=(512, 512,3), 
                 img_dir=TRAIN_IMAGES_DIR, n_classes = 4, train =True,
                 n_augment = 9, shuffle=True,
                 *args, **kwargs):

        self.list_IDs = list_IDs
        self.indices = np.arange(len(self.list_IDs))
        self.img_labels = img_labels ### contains col1: names of images for loading + col2(!exits fr test) for labels
        self.n_classes = n_classes   ### nb of classes
        self.n_augment = n_augment   ### nb of additional data samples
        self.batch_size = batch_size
        self.img_size = img_size    ###  desired image size: (width, height, n_channels)
        self.img_dir = img_dir
        self.shuffle = shuffle
        self.train = train
        self.on_epoch_end()

    def __len__(self):
        return int(ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size: (index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]
        if self.train:
            X, Y = self.__data_generation(list_IDs_temp)
            return X, Y
        else:
            X = self.__data_generation(list_IDs_temp)
            return X
        
    def on_epoch_end(self):
        self.indices = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, list_IDs_temp):
        print("Self image size",self.img_size )
        X = np.empty((self.batch_size * (self.n_augment + 1), *self.img_size))
        
        if self.train: # training phase
            Y = np.zeros((self.batch_size * (self.n_augment + 1), self.n_classes),
                          dtype=np.float32)
        
            for i, ID in enumerate(list_IDs_temp):
              test = _read(self.img_dir+self.img_labels['ID'].loc[ID] +".jpg",
                              self.img_size,  augment_data= self.n_augment, plot=False)
              print("Dim data gen",  test.shape)
              X[i:(i +self.n_augment + 1),] = test
              ### Convert label  into one hot vector
              Y[i:(i +self.n_augment + 1), int(self.img_labels['label'].loc[ID])] = 1

            return X, Y
        
        else: # test phase
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = _read(self.img_dir+self.img_labels['ID'].loc[ID] +".jpg",
                              self.img_size, augment_data=0, plot=False)      
            return X


def _normalize(img):
    if img.max() == img.min():
        return np.zeros(img.shape)-1
    return 2 * (img - img.min())/(img.max() - img.min()) - 1


  
def _read(path, desired_size, augment_data=0, plot=False):
    """Will be used in DataGenerator
    Loads image, crops and resizes. With optional image data augmentation.
    We assume that the image has been centered. 
    Input:
    ----------------------------------
    desired_size     :  desired size for the image (tuple)
    augment_data     :  nb of data augmented samples (int)
    """
    new_width, new_height,_ = desired_size
    print (path)
    img = cv2.imread(path)
    rows, cols,_ = img.shape
    if rows < cols:  
      M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
      img = cv2.warpAffine(img,M,(cols,rows))
    res = cv2.resize(img, dsize=desired_size[:2], interpolation=cv2.INTER_CUBIC)
    samples = np.expand_dims(res, 0)
    if augment_data>0:
        # create image data augmentation generator
        datagen = ImageDataGenerator(rotation_range=90,
                                     width_shift_range=[-100,100])
        # prepare iterator
        it = datagen.flow(samples, batch_size=1)
        for i in range(augment_data):
          batch = it.next()    # generate batch of images
          image = batch[0]
          samples = np.vstack((samples, np.expand_dims(image, 0)))
          if plot:
            plt.subplot(330 + 1 + i)
            plt.imshow(batch[0].astype('uint8'))
          #img = np.stack((res,)*3, axis=-1)
    if plot: plt.show()
    print('samples size in read:', samples.shape)
    return samples



def weighted_log_loss(y_true, y_pred):
    """
    Can be used as the loss function in model.compile()
    ---------------------------------------------------
    """
    
    class_weights = np.array([1., 2., 2., 2.])
    
    eps = K.epsilon()
    
    y_pred = K.clip(y_pred, eps, 1.0-eps)

    out = -(         y_true  * K.log(      y_pred) * class_weights
            + (1.0 - y_true) * K.log(1.0 - y_pred) * class_weights)
    
    return K.mean(out, axis=-1)


def _normalized_weighted_average(arr, weights=None):
    """
    A simple Keras implementation that mimics that of 
    numpy.average(), specifically for the this competition
    """
    
    if weights is not None:
        scl = K.sum(weights)
        weights = K.expand_dims(weights, axis=1)
        return K.sum(K.dot(arr, weights), axis=1) / scl
    return K.mean(arr, axis=1)


def weighted_loss(y_true, y_pred):
    """
    Will be used as the metric in model.compile()
    ---------------------------------------------
    
    Similar to the custom loss function 'weighted_log_loss()' above
    but with normalized weights, which should be very similar 
    to the official competition metric:
        https://www.kaggle.com/kambarakun/lb-probe-weights-n-of-positives-scoring
    and hence:
        sklearn.metrics.log_loss with sample weights
    """
    
    class_weights = K.variable([1., 2., 2., 2.])
    
    eps = K.epsilon()
    
    y_pred = K.clip(y_pred, eps, 1.0-eps)

    loss = -(        y_true  * K.log(      y_pred)
            + (1.0 - y_true) * K.log(1.0 - y_pred))
    
    loss_samples = _normalized_weighted_average(loss, class_weights)
    
    return K.mean(loss_samples)


def weighted_log_loss_metric(trues, preds):
    """
    Will be used to calculate the log loss 
    of the validation set in PredictionCheckpoint()
    ------------------------------------------
    """
    class_weights = [1., 2., 2., 2.]
    
    epsilon = 1e-7
    
    preds = np.clip(preds, epsilon, 1-epsilon)
    loss = trues * np.log(preds) + (1 - trues) * np.log(1 - preds)
    loss_samples = np.average(loss, axis=1, weights=class_weights)

    return - loss_samples.mean()

def sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1):
    return K.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=from_logits, axis=axis)


# TEST IMG READ
# test = _read('trimg/image1.jpg',(512,512,3),9, plot=True)
# print(test.shape)

def read_testset(filename="trimg/classification_labels.txt"):
    ''' 
    Data in data folder
    '''
    df = pd.read_csv(filename, sep=" ", header=None)
    df.columns = ["ID", "label"]
    return df

def read_trainset(filename="trimg/classification_labels.txt"):
    df = pd.read_csv(filename, sep=" ", header=None)
    df.columns = ["ID", "label"]
    return df
