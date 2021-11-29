import os
import keras
import math
import pandas as pd
import numpy

from datetime import datetime
from datetime import date
from model import lanenet

from datagen import DataGenerator

# Standard imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

# from define import global_define
import myconfig

# fix random seed for reproducibility
numpy.random.seed(7)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

local_epoch = [100]
step = [0]
top_crop = [0.5]
bottom_crop = [1.0]

# for i in range(len(local_epoch)):
# for i in range(len(top_crop)):
for i in range (len(step)):
    starttime = datetime.now()
    #today = datetime.now().strftime('%Y%m%d%H%M')
    #filename = "history/history_%s.txt" % (today)
    #fhistory = open(filename, 'w')

    #print("start time : ", starttime)
    data = pd.read_csv(myconfig.train_datapath)
    val_data = pd.read_csv(myconfig.val_datapath)

    steps_per_epoch = math.ceil(len(data) / 64)
    validation_steps = math.ceil(len(val_data) / 64)

    # Parameters
    train_params = {'dim': (myconfig.resize_image_width, myconfig.resize_image_height),
              'batch_size': 64,
              'n_classes': myconfig.num_classes,
              'n_channels': 3,
              'top_percent' : top_crop[i],
              'bottom_percent' : bottom_crop[i],
              'shuffle': True}

    val_params = {'dim': (myconfig.resize_image_width, myconfig.resize_image_height),
              'batch_size': 64,
              'n_classes': myconfig.num_classes,
              'n_channels': 3,
              'top_percent' : top_crop[i],
              'bottom_percent' : bottom_crop[i],
              'shuffle': True}

    # Generators
    train_generator = DataGenerator(data, **train_params)
    validation_generator = DataGenerator(val_data, **val_params)
    tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True, update_freq=128)

    # Save the model according to the conditions
    # with tf.device("/cpu:0"):
    model1 = lanenet.LaneNet_NV7_new.build(width=myconfig.resize_image_width, height=myconfig.resize_image_height, depth=3, classes=myconfig.num_classes)
    adam = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=True)
    model1.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics=['acc'])

    # modelname = "model_%s_%s_%s_%s_full.h5" % (today, myconfig.resize_image_width, myconfig.resize_image_height, local_epoch[i])
    checkpoint_filepath="./checkpoint/weights-improvement-{epoch:02d}-{loss:.4f}-{val_loss:.4f}-{val_acc:.4f}.hdf5"
    checkpoint= ModelCheckpoint(checkpoint_filepath, verbose = 0, monitor='val_loss', save_best_only = False, save_weights_only = False, mode = "min",  period = 1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

    # callbacks_list = [tb_hist]
    callbacks_list = [tb_hist, early]
# callbacks_list = [tb_hist, checkpoint]
    # callbacks_list = [tb_hist, checkpoint, early]

    history = model1.fit_generator(train_generator,
                                      steps_per_epoch = steps_per_epoch,
                                      validation_data = validation_generator,
                                      validation_steps = validation_steps,
                                      use_multiprocessing=True,
                                      epochs = local_epoch[0], workers=8, verbose = 1, callbacks=callbacks_list)


    modelname = ('./checkpoint/model-{}-{}-loss:{loss:.4f}-val_loss:{val_loss:.4f}-val_acc:{val_acc:.4f}.hdf5'.format(today, history.epoch[-1], loss=history.history['loss'][-11], val_loss=history.history['val_loss'][-11], val_acc=history.history['val_acc'][-11]))

    model1.save(modelname)
    # model1.save(checkpoint_filepath)
    endtime = datetime.now()
    print("end time : ", endtime-starttime)

    # log = "%s, %s, %s\n" % (history.history['loss'], history.history['val_loss'], (endtime-starttime))
    # fhistory.write(log)
    # fhistory.close()
