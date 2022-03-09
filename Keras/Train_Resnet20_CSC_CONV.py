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

    x = AveragePooling2D(pool_size=8, data_format = 'channels_last')(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)
    model = Model(inputs=inputs, outputs=outputs)
    return model




model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule(0)), metrics=['accuracy'])
model.summary()
print(model_type)

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

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
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0)

    datagen.fit(x_train)

    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                       steps_per_epoch = 1563,
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])





