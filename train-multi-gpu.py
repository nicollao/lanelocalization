from datetime import datetime
from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Activation
from keras.layers import Dropout
from keras.callbacks import LearningRateScheduler
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.optimizers import Adadelta
from keras.utils import Sequence
from datetime import date
import numpy
import keras

from my_classes import DataGenerator
import pandas as pd

# Standard imports
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model

# from define import global_define
import myconfig
import imagedefine

# learning rate schedule
def step_decay(epoch):
    print('input : ', epoch)
    if epoch >= 40:
        lrate = 0.0001
    elif epoch >= 40:
        lrate = 0.0003
    else:
        lrate = 0.0005
    # initial_lrate = 0.0005
    # drop = 0.0004
    # epochs_drop = 10.0
    # lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    print(lrate)
    return lrate

# fix random seed for reproducibility
numpy.random.seed(7)


# origin netwnork model
if myconfig._USE_GPU_COUNT_ > 1:
    with tf.device("/cpu:0"):
        model = Sequential()
else:
    model = Sequential()

model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape = (myconfig.resize_image_height, myconfig.resize_image_width, 3)))

# convolution layer : input image size
model.add(Conv2D(24, (5, 5), padding = 'same', strides = (2, 2), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (1, 1)))
model.add(Dropout(0.25))
model.add(Conv2D(36, (5, 5), padding = 'same', strides = (2, 2), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (1, 1)))
model.add(Dropout(0.25))
model.add(Conv2D(48, (5, 5), padding = 'same', strides = (2, 2), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (1, 1)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (1, 1)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (1, 1)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))

if myconfig.def_output_type == 2:
    model.add(Dense(myconfig.num_classes))
    model.add(Activation('softmax'))
    if myconfig._USE_GPU_COUNT_ > 1:
        multi_model = multi_gpu_model(model, gpus=myconfig._USE_GPU_COUNT_)
        multi_model.compile(loss = 'categorical_crossentropy',  optimizer='adadelta', metrics=['accuracy'])

    # model.compile(loss = 'categorical_crossentropy',  optimizer=Adadelta(1e-4), metrics=['accuracy'])
    # multi_model.compile(loss = 'categorical_crossentropy',  optimizer=Adadelta(1e-4), metrics=['accuracy'])
    # adadelta == Adaptive Delta
    # keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    model.compile(loss = 'categorical_crossentropy',  optimizer='adadelta', metrics=['accuracy'])

else:
    model.add(Dense(1))
    model.add(Activation('relu'))
    model.summary()
    multi_model = multi_gpu_model(model, gpus=2)
    model.compile(loss = 'mse', optimizer = Adam(1e-4))
    multi_model.compile(loss = 'mse', optimizer = Adam(1e-4))


local_epoch = [100]
today = date.today().strftime('%Y%m%d')
filename = "history_%s.txt" % (today)
fhistory = open(filename, 'w')
fhistory.close()

#lrate = LearningRateScheduler(step_decay)
# def step_decay(epoch):
#    initial_lrate = 0.0005
#    drop = 0.0002
#    epochs_drop = 10.0
#    print(math.floor((1+epoch)/epochs_drop))
#    print(math.pow(drop, math.floor((1+epoch)/epochs_drop)))
#    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
#
#     print('lrate :: ', lrate)
#     return lrate


for i in range(len(local_epoch)):
    fhistory = open(filename, 'a')
    starttime = datetime.now()
    print("start time : ")
    print(starttime)

    # imagedefine.init_imagedefine()
    data = pd.read_csv(myconfig.train_datapath)
    val_data = pd.read_csv(myconfig.val_datapath)

    # Parameters
    train_params = {'dim': (myconfig.resize_image_width, myconfig.resize_image_height),
              'batch_size': 64,
              'n_classes': myconfig.num_classes,
              'n_channels': 3,
              'shuffle': True}

    val_params = {'dim': (myconfig.resize_image_width, myconfig.resize_image_height),
              'batch_size': 64,
              'n_classes': myconfig.num_classes,
              'n_channels': 3,
              'shuffle': True}

    # Generators
    train_generator = DataGenerator(data, **train_params)
    validation_generator = DataGenerator(val_data, **val_params)

    # train_generator = imagedefine.generate_next_batch(64, 0, 0)
    # validation_generator = imagedefine.generate_next_batch(64, 0, 1)

    # tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=2, write_graph=True, write_images=True, update_freq='batch')
    tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True, update_freq=64)

    if myconfig._USE_GPU_COUNT_ > 1:
        history = multi_model.fit_generator(train_generator, steps_per_epoch = myconfig.steps_per_epoch,
                                        validation_data = validation_generator,
                                        validation_steps = myconfig.validation_steps,
                                        epochs = local_epoch[i], verbose = 1, callbacks=[tb_hist])
    else:
        history = model.fit_generator(train_generator,
                                      steps_per_epoch = myconfig.steps_per_epoch,
                                      validation_data = validation_generator,
                                      validation_steps = myconfig.validation_steps,
                                      use_multiprocessing=True,
                                      epochs = local_epoch[i], workers=8, verbose = 1, callbacks=[tb_hist])


    print(history.history['loss'])
    print(history.history['val_loss'])

    modelname = "model_%s_%s_%s_%s_full.h5" % (today, myconfig.resize_image_width, myconfig.resize_image_height, local_epoch[i])
    # modelname = "model.h5"
    if myconfig._USE_GPU_COUNT_ > 1:
        model.set_weights(multi_model.get_weights())

    model.save(modelname)

    # model.save(myconfig.modelname)
    endtime = datetime.now()
    print("end time : ", endtime-starttime)

    log = "%s, %s, %s\n" % (history.history['loss'], history.history['val_loss'], (endtime-starttime))
    fhistory.write(log)
    fhistory = open(filename, 'a')
