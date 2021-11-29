from datetime import datetime
from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from datetime import date


# from define import global_define
import myconfig
import imagedefine

import tensorflow as tf
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

# K._LEARNING_PHASE = tf.constant(0) # try with 1
# K.set_learning_phase(1) #set learning phase

# origin network model
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
model.add(Dense(1))
model.summary()
model.compile(loss = 'mse', optimizer = Adam(1e-4))


local_epoch = [10]
# local_epoch = [1, 2, 3]
today = date.today().strftime('%Y%m%d')
filename = "history_%s.txt" % (today)
fhistory = open(filename, 'w')
fhistory.close()

for i in range(len(local_epoch)):
    # for i in range(len(local_epoch)):
    fhistory = open(filename, 'a')
    starttime = datetime.now()
    print("start time : ")
    print(starttime)

    train_generator = imagedefine.generate_next_batch()
    validation_generator = imagedefine.generate_next_batch()

    check1 = datetime.now()
    print("check1 time : ")
    print(check1 - starttime)

    history = model.fit_generator(train_generator, steps_per_epoch = myconfig.steps_per_epoch, validation_data = validation_generator, validation_steps = myconfig.validation_steps, epochs = local_epoch[i], verbose = 1, use_multiprocessing=True, max_queue_size=1024)

    check2 = datetime.now()
    print("check2 time : ")
    print(check2 - check1)

    print(history.history['loss'])
    print(history.history['val_loss'])

    modelname = "model_%s_%s_%s_%s_full.h5" % (today, myconfig.resize_image_width, myconfig.resize_image_height, local_epoch[i])
    # modelname = "model.h5"
    model.save(modelname)
    # model.save(myconfig.modelname)
    endtime = datetime.now()
    print("end time : ", endtime-starttime)

    log = "%s, %s, %s\n" % (history.history['loss'], history.history['val_loss'], (endtime-starttime))
    fhistory.write(log)
    fhistory = open(filename, 'a')
