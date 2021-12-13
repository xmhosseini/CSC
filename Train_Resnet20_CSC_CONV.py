"""
#Trains a ResNet on the CIFAR10 dataset.
ResNet v1:
[Deep Residual Learning for Image Recognition
](https://arxiv.org/pdf/1512.03385.pdf)
ResNet v2:
[Identity Mappings in Deep Residual Networks
](https://arxiv.org/pdf/1603.05027.pdf)
Model|n|200-epoch accuracy|Original paper accuracy |sec/epoch GTX1080Ti
:------------|--:|-------:|-----------------------:|---:
ResNet20   v1|  3| 92.16 %|                 91.25 %|35
ResNet32   v1|  5| 92.46 %|                 92.49 %|50
ResNet44   v1|  7| 92.50 %|                 92.83 %|70
ResNet56   v1|  9| 92.71 %|                 93.03 %|90
ResNet110  v1| 18| 92.65 %|            93.39+-.16 %|165
ResNet164  v1| 27|     - %|                 94.07 %|  -
ResNet1001 v1|N/A|     - %|                 92.39 %|  -
&nbsp;
Model|n|200-epoch accuracy|Original paper accuracy |sec/epoch GTX1080Ti
:------------|--:|-------:|-----------------------:|---:
ResNet20   v2|  2|     - %|                     - %|---
ResNet32   v2|N/A| NA    %|            NA         %| NA
ResNet44   v2|N/A| NA    %|            NA         %| NA
ResNet56   v2|  6| 93.01 %|            NA         %|100
ResNet110  v2| 12| 93.15 %|            93.63      %|180
ResNet164  v2| 18|     - %|            94.54      %|  -
ResNet1001 v2|111|     - %|            95.08+-.14 %|  -
"""

from __future__ import print_function
import keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, DepthwiseConv2D, Lambda, Concatenate
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from keras import backend as K
# from keras.models import Model
from tensorflow.keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os


import tensorflow as tf
from MyConv2_K import my_conv2d

# keras.backend.set_image_data_format('channels_last')
keras.backend.set_image_data_format('channels_last')
data_format = 'channels_last'
# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 200
data_augmentation = True
num_classes = 10

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n = 3

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1

# Computed depth from supplied model parameter n
depth = 20


# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]




# img_rows, img_cols = 32, 32
# if K.image_data_format() == 'channels_first':
    # x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    # x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    # input_shape = (3, img_rows, img_cols)
# else:
    # x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    # x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    # input_shape = (img_rows, img_cols, 3)




# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def getCustomTensor(H, W, C, N, F, S):
  CustomTensor = np.full((H, W, C, N), 0)
  for f in range (0, F):
    for w in range (0, W):
      for h in range (0, H):
        CustomTensor [h, w, 0, f*S]= 1
  for c in range (1, C):
    for n in range (0, N):
      for w in range (0, W):
        for h in range (0, H):
          CustomTensor [h, w, c, n] = CustomTensor [h, w, c-1, (n-1)%N]
  return CustomTensor


def getCustomMatrix(C, N, F, S):
  CustomMatrix = np.full((C, N), 0)
  for f in range (0, F):
    CustomMatrix [0, f*S]= 1
  for c in range (1, C):
    for n in range (0, N):
      CustomMatrix [c, n] = CustomMatrix [c-1, (n-1)%N]
  return CustomMatrix


def resnet_layer_first(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True):
    conv = Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), data_format = 'channels_last')
    x = inputs
    x = conv(x)
    if batch_normalization:
        x = BatchNormalization(axis=-1)(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x

def resnet_layer_16x16(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True):
    x = inputs
    x = my_conv2d(inputs=x,filters= 16, my_filter = getCustomTensor(1, 3, 16, 16, 8, 1), kernel_size=(1,3),use_bias=None, strides=(1,1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), data_format = 'channels_last')
    x = my_conv2d(inputs=x,filters= 16, my_filter = getCustomTensor(3, 1, 16, 16, 4, 4), kernel_size=(3,1), strides=(1,1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), data_format = 'channels_last')
    if batch_normalization:
        x = BatchNormalization(axis=-1)(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x


def resnet_layer_16x32(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True):
    x = inputs
    if kernel_size > 1:
        x = my_conv2d(inputs=x,filters= 16, my_filter = getCustomTensor(1, kernel_size, 16, 16, 4, 1), kernel_size=(1,kernel_size),use_bias=None, strides=(1,1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), data_format = 'channels_last')
        x = my_conv2d(inputs=x,filters= 32, my_filter = getCustomTensor(kernel_size, 1, 16, 32, 8, 4), kernel_size=(kernel_size,1), strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), data_format = 'channels_last')
    else:
        x = my_conv2d(inputs=x,filters= 32, my_filter = getCustomTensor(1, 1, 16, 32, 32, 1), kernel_size=(1,1), strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), data_format = 'channels_last')
    if batch_normalization:
        x = BatchNormalization(axis=-1)(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x

def resnet_layer_32x32(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True):
    x = inputs
    x = my_conv2d(inputs=x,filters= 32, my_filter = getCustomTensor(1, 3, 32, 32, 8, 1), kernel_size=(1,3),use_bias=None, strides=(1,1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), data_format = 'channels_last')
    x = my_conv2d(inputs=x,filters= 32, my_filter = getCustomTensor(3, 1, 32, 32, 4, 8), kernel_size=(3,1), strides=(1,1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), data_format = 'channels_last')
    if batch_normalization:
        x = BatchNormalization(axis=-1)(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x

def resnet_layer_32x64(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True):
    x = inputs
    if kernel_size > 1:
        x = my_conv2d(inputs=x,filters= 32, my_filter = getCustomTensor(1, kernel_size, 32, 32, 8, 1), kernel_size=(1,kernel_size),use_bias=None, strides=(1,1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), data_format = 'channels_last')
        x = my_conv2d(inputs=x,filters= 64, my_filter = getCustomTensor(kernel_size, 1, 32, 64, 8, 8), kernel_size=(kernel_size,1), strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), data_format = 'channels_last')
    else:
        x = my_conv2d(inputs=x,filters= 64, my_filter = getCustomTensor(1, 1, 32, 64, 64, 1), kernel_size=(1,1), strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), data_format = 'channels_last')
    if batch_normalization:
        x = BatchNormalization(axis=-1)(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x

def resnet_layer_64x64(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True):
    x = inputs
    x = my_conv2d(inputs=x,filters= 64, my_filter = getCustomTensor(1, 3, 64, 64, 8, 1), kernel_size=(1,3),use_bias=None, strides=(1,1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), data_format = 'channels_last')
    x = my_conv2d(inputs=x,filters= 64, my_filter = getCustomTensor(3, 1, 64, 64, 8, 8), kernel_size=(3,1), strides=(1,1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), data_format = 'channels_last')
    if batch_normalization:
        x = BatchNormalization(axis=-1)(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x



def resnet_v1(input_shape, depth, num_classes=10):

    inputs = Input(shape=input_shape)
    x = resnet_layer_first(inputs=inputs)
    # Instantiate the stack of residual units

    strides = 1
    y = resnet_layer_16x16(inputs=x, num_filters=16, strides=strides)
    y = resnet_layer_16x16(inputs=y, num_filters=16, activation=None)
    print (x.shape, y.shape, '*************************************')
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)
    print(x.shape,'******************')
    y = resnet_layer_16x16(inputs=x, num_filters=16, strides=strides)
    y = resnet_layer_16x16(inputs=y, num_filters=16, activation=None)
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)

    y = resnet_layer_16x16(inputs=x, num_filters=16, strides=strides)
    y = resnet_layer_16x16(inputs=y, num_filters=16, activation=None)
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)



    strides = 2  # downsample
    y = resnet_layer_16x32(inputs=x, num_filters=32, strides=strides)
    y = resnet_layer_32x32(inputs=y, num_filters=32, activation=None)
    x = resnet_layer_16x32(inputs=x, num_filters=32, kernel_size=1, strides=strides, activation=None, batch_normalization=False)
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)

    strides = 1
    y = resnet_layer_32x32(inputs=x, num_filters=32, strides=strides)
    y = resnet_layer_32x32(inputs=y, num_filters=32, activation=None)
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)

    y = resnet_layer_32x32(inputs=x, num_filters=32, strides=strides)
    y = resnet_layer_32x32(inputs=y, num_filters=32, activation=None)
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)



    strides = 2  # downsample
    y = resnet_layer_32x64(inputs=x, num_filters=64, strides=strides)
    y = resnet_layer_64x64(inputs=y, num_filters=64, activation=None)
    x = resnet_layer_32x64(inputs=x, num_filters=64, kernel_size=1, strides=strides, activation=None, batch_normalization=False)
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)

    strides = 1
    y = resnet_layer_64x64(inputs=x, num_filters=64, strides=strides)
    y = resnet_layer_64x64(inputs=y, num_filters=64, activation=None)
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)

    y = resnet_layer_64x64(inputs=x, num_filters=64, strides=strides)
    y = resnet_layer_64x64(inputs=y, num_filters=64, activation=None)
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)



    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8, data_format = 'channels_last')(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model




model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule(0)), metrics=['accuracy'])
model.summary()
print(model_type)

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                       steps_per_epoch = 1563,
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)
# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


















# from __future__ import print_function
# import keras
# from keras.layers import Dense, Conv2D, BatchNormalization, Activation, DepthwiseConv2D, Lambda, Concatenate
# from keras.layers import AveragePooling2D, Input, Flatten
# from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from keras.callbacks import ReduceLROnPlateau
# from keras.preprocessing.image import ImageDataGenerator
# from keras.regularizers import l2
# from keras import backend as K
# from keras.models import Model
# from keras.datasets import cifar10
# import numpy as np
# import os





# # Training parameters
# batch_size = 32  # orig paper trained all networks with batch_size=128
# epochs = 200*1.5
# data_augmentation = True
# num_classes = 10

# # Subtracting pixel mean improves accuracy
# subtract_pixel_mean = True

# # Model parameter
# # ----------------------------------------------------------------------------
# #           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# # Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
# #           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# # ----------------------------------------------------------------------------
# # ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# # ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# # ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# # ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# # ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# # ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# # ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# # ---------------------------------------------------------------------------
# n = 3

# # Model version
# # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
# version = 1

# # Computed depth from supplied model parameter n
# depth = 20


# # Model name, depth and version
# model_type = 'ResNet%dv%d' % (depth, version)

# # Load the CIFAR10 data.
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()

# # Input image dimensions.
# input_shape = x_train.shape[1:]

# # Normalize data.
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255

# # If subtract pixel mean is enabled
# if subtract_pixel_mean:
    # x_train_mean = np.mean(x_train, axis=0)
    # x_train -= x_train_mean
    # x_test -= x_train_mean

# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
# print('y_train shape:', y_train.shape)

# # Convert class vectors to binary class matrices.
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)


# def lr_schedule(epoch):
    # """Learning Rate Schedule
    # Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    # Called automatically every epoch as part of callbacks during training.
    # # Arguments
        # epoch (int): The number of epochs
    # # Returns
        # lr (float32): learning rate
    # """
    # lr = 1e-3
    # if epoch > 180*1.5:
        # lr *= 0.5e-3
    # elif epoch > 160*1.5:
        # lr *= 1e-3
    # elif epoch > 120*1.5:
        # lr *= 1e-2
    # elif epoch > 80*1.5:
        # lr *= 1e-1
    # print('Learning rate: ', lr)
    # return lr




# def F_DepthwiseConv2D(inputs,
		# F=2,
		# S = 1,
		# depth_multiplier = 1,
		# kernel_size = (3,3),
		# strides=(1, 1),
		# padding='same',
		# use_bias=True,
		# kernel_regularizer=l2(1e-4)):

	# output= DepthwiseConv2D(kernel_size=kernel_size, strides=strides,use_bias=use_bias, padding=padding,depth_multiplier=depth_multiplier,kernel_regularizer=kernel_regularizer)(inputs)
	# for i in range(1,F):
		# depth_append = []
		# depth = DepthwiseConv2D(kernel_size=kernel_size, strides=strides,use_bias=None, padding=padding,depth_multiplier=depth_multiplier,kernel_regularizer=kernel_regularizer)(inputs)
		# depth_1= Lambda (lambda x: x[:,:,:,i*S:])(depth);
		# depth_2= Lambda (lambda x: x[:,:,:,0:i*S])(depth);
		# depth_append.append(depth_1);
		# depth_append.append(depth_2);
		# depth = Concatenate(axis=3)(depth_append);
		# output = keras.layers.add([output, depth])
	# return output



# def resnet_layer_first(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True):
    # conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))
    # x = inputs
    # x = conv(x)
    # if batch_normalization:
        # x = BatchNormalization()(x)
    # if activation is not None:
        # x = Activation(activation)(x)
    # return x

# def resnet_layer_16x16(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True):
    # x = inputs
    # x = F_DepthwiseConv2D(x,F=4, S = 1, kernel_size=(1,3),use_bias=None, strides=(1,1), padding='same')
    # x = F_DepthwiseConv2D(x,F=4, S = 4, kernel_size=(3,1), strides=(1,1), padding='same')
    # if batch_normalization:
        # x = BatchNormalization()(x)
    # if activation is not None:
        # x = Activation(activation)(x)
    # return x

# def resnet_layer_16x32(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True):
    # x = inputs
    # x = F_DepthwiseConv2D(x,F=4, S = 1, kernel_size=(1,kernel_size),use_bias=None, strides=(1,1), padding='same')
    # x1= F_DepthwiseConv2D(x,F=4, S = 4, kernel_size=(kernel_size,1), strides=strides, padding='same')
    # x2= F_DepthwiseConv2D(x,F=4, S = 4, kernel_size=(kernel_size,1), strides=strides, padding='same')
    # x=Concatenate(axis=3)([x1,x2])
    # if batch_normalization:
        # x = BatchNormalization()(x)
    # if activation is not None:
        # x = Activation(activation)(x)
    # return x

# def resnet_layer_32x32(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True):
    # x = inputs
    # x = F_DepthwiseConv2D(x,F=8, S = 1, kernel_size=(1,3),use_bias=None, strides=(1,1), padding='same')
    # x = F_DepthwiseConv2D(x,F=4, S = 8, kernel_size=(3,1), strides=(1,1), padding='same')
    # if batch_normalization:
        # x = BatchNormalization()(x)
    # if activation is not None:
        # x = Activation(activation)(x)
    # return x

# def resnet_layer_32x64(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True):
    # x = inputs
    # x = F_DepthwiseConv2D(x,F=8, S = 1, kernel_size=(1,kernel_size),use_bias=None, strides=(1,1), padding='same')
    # x1= F_DepthwiseConv2D(x,F=4, S = 8, kernel_size=(kernel_size,1), strides=strides, padding='same')
    # x2= F_DepthwiseConv2D(x,F=4, S = 8, kernel_size=(kernel_size,1), strides=strides, padding='same')
    # x=Concatenate(axis=3)([x1,x2])
    # if batch_normalization:
        # x = BatchNormalization()(x)
    # if activation is not None:
        # x = Activation(activation)(x)
    # return x

# def resnet_layer_64x64(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True):
    # x = inputs
    # x = F_DepthwiseConv2D(x,F=8, S = 1, kernel_size=(1,3),use_bias=None, strides=(1,1), padding='same')
    # x = F_DepthwiseConv2D(x,F=8, S = 8,kernel_size=(3,1), strides=(1,1), padding='same')

    # if batch_normalization:
        # x = BatchNormalization()(x)
    # if activation is not None:
        # x = Activation(activation)(x)
    # return x



# def resnet_v1(input_shape, depth, num_classes=10):

    # inputs = Input(shape=input_shape)
    # x = resnet_layer_first(inputs=inputs)
    # # Instantiate the stack of residual units

    # strides = 1
    # y = resnet_layer_16x16(inputs=x, num_filters=16, strides=strides)
    # y = resnet_layer_16x16(inputs=y, num_filters=16, activation=None)
    # x = keras.layers.add([x, y])
    # x = Activation('relu')(x)

    # y = resnet_layer_16x16(inputs=x, num_filters=16, strides=strides)
    # y = resnet_layer_16x16(inputs=y, num_filters=16, activation=None)
    # x = keras.layers.add([x, y])
    # x = Activation('relu')(x)

    # y = resnet_layer_16x16(inputs=x, num_filters=16, strides=strides)
    # y = resnet_layer_16x16(inputs=y, num_filters=16, activation=None)
    # x = keras.layers.add([x, y])
    # x = Activation('relu')(x)



    # strides = 2  # downsample
    # y = resnet_layer_16x32(inputs=x, num_filters=32, strides=strides)
    # y = resnet_layer_32x32(inputs=y, num_filters=32, activation=None)
    # x = resnet_layer_16x32(inputs=x, num_filters=32, kernel_size=1, strides=strides, activation=None, batch_normalization=False)
    # x = keras.layers.add([x, y])
    # x = Activation('relu')(x)

    # strides = 1
    # y = resnet_layer_32x32(inputs=x, num_filters=32, strides=strides)
    # y = resnet_layer_32x32(inputs=y, num_filters=32, activation=None)
    # x = keras.layers.add([x, y])
    # x = Activation('relu')(x)

    # y = resnet_layer_32x32(inputs=x, num_filters=32, strides=strides)
    # y = resnet_layer_32x32(inputs=y, num_filters=32, activation=None)
    # x = keras.layers.add([x, y])
    # x = Activation('relu')(x)



    # strides = 2  # downsample
    # y = resnet_layer_32x64(inputs=x, num_filters=64, strides=strides)
    # y = resnet_layer_64x64(inputs=y, num_filters=64, activation=None)
    # x = resnet_layer_32x64(inputs=x, num_filters=64, kernel_size=1, strides=strides, activation=None, batch_normalization=False)
    # x = keras.layers.add([x, y])
    # x = Activation('relu')(x)

    # strides = 1
    # y = resnet_layer_64x64(inputs=x, num_filters=64, strides=strides)
    # y = resnet_layer_64x64(inputs=y, num_filters=64, activation=None)
    # x = keras.layers.add([x, y])
    # x = Activation('relu')(x)

    # y = resnet_layer_64x64(inputs=x, num_filters=64, strides=strides)
    # y = resnet_layer_64x64(inputs=y, num_filters=64, activation=None)
    # x = keras.layers.add([x, y])
    # x = Activation('relu')(x)



    # # Add classifier on top.
    # # v1 does not use BN after last shortcut connection-ReLU
    # x = AveragePooling2D(pool_size=8)(x)
    # y = Flatten()(x)
    # outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)

    # # Instantiate model.
    # model = Model(inputs=inputs, outputs=outputs)
    # return model




# model = resnet_v1(input_shape=input_shape, depth=depth)

# model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule(0)), metrics=['accuracy'])
# model.summary()
# print(model_type)

# # Prepare model model saving directory.
# save_dir = os.path.join(os.getcwd(), 'saved_models')
# model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
# if not os.path.isdir(save_dir):
    # os.makedirs(save_dir)
# filepath = os.path.join(save_dir, model_name)

# # Prepare callbacks for model saving and for learning rate adjustment.
# checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True)

# lr_scheduler = LearningRateScheduler(lr_schedule)

# lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

# callbacks = [checkpoint, lr_reducer, lr_scheduler]

# # Run training, with or without data augmentation.
# if not data_augmentation:
    # print('Not using data augmentation.')
    # model.fit(x_train, y_train,
              # batch_size=batch_size,
              # epochs=epochs,
              # validation_data=(x_test, y_test),
              # shuffle=True,
              # callbacks=callbacks)
# else:
    # print('Using real-time data augmentation.')
    # # This will do preprocessing and realtime data augmentation:
    # datagen = ImageDataGenerator(
        # # set input mean to 0 over the dataset
        # featurewise_center=False,
        # # set each sample mean to 0
        # samplewise_center=False,
        # # divide inputs by std of dataset
        # featurewise_std_normalization=False,
        # # divide each input by its std
        # samplewise_std_normalization=False,
        # # apply ZCA whitening
        # zca_whitening=False,
        # # epsilon for ZCA whitening
        # zca_epsilon=1e-06,
        # # randomly rotate images in the range (deg 0 to 180)
        # rotation_range=0,
        # # randomly shift images horizontally
        # width_shift_range=0.1,
        # # randomly shift images vertically
        # height_shift_range=0.1,
        # # set range for random shear
        # shear_range=0.,
        # # set range for random zoom
        # zoom_range=0.,
        # # set range for random channel shifts
        # channel_shift_range=0.,
        # # set mode for filling points outside the input boundaries
        # fill_mode='nearest',
        # # value used for fill_mode = "constant"
        # cval=0.,
        # # randomly flip images
        # horizontal_flip=True,
        # # randomly flip images
        # vertical_flip=False,
        # # set rescaling factor (applied before any other transformation)
        # rescale=None,
        # # set function that will be applied on each input
        # preprocessing_function=None,
        # # image data format, either "channels_first" or "channels_last"
        # data_format=None,
        # # fraction of images reserved for validation (strictly between 0 and 1)
        # validation_split=0.0)

    # # Compute quantities required for featurewise normalization
    # # (std, mean, and principal components if ZCA whitening is applied).
    # datagen.fit(x_train)

    # # Fit the model on the batches generated by datagen.flow().
    # model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        # validation_data=(x_test, y_test),
                        # epochs=epochs, verbose=1, workers=4,
                        # callbacks=callbacks)

# # Score trained model.
# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])

