# 이미지 처리 분야에서 가장 유명한 신경망 모델인 CNN 을 이용하여 더 높은 인식률을 만들어봅니다.
import tensorflow as tf

# from define import global_define
import myconfig
import imagedefine

X = tf.placeholder(tf.uint8, [None, 200, 66, 3])
Y = tf.placeholder(tf.float32, [None, 1])
keep_prob = tf.placeholder(tf.uint8)

# 각각의 변수와 레이어는 다음과 같은 형태로 구성됩니다.
# W1 [5 5 1 24] -> [5 5]: 커널 크기, 1: 입력값 X 의 특성수, 24: 필터 갯수
# L1 Conv shape=(?, 66, 66, 24)
#    Pool     ->(?, 33, 33, 24)
W1 = tf.Variable(tf.random_normal([5, 5, 1, 24], stddev=0.01))
# tf.nn.conv2d 를 이용해 한칸씩 움직이는 컨볼루션 레이어를 쉽게 만들 수 있습니다.
# padding='SAME' 은 커널 슬라이딩시 최외곽에서 한칸 밖으로 더 움직이는 옵션
L1 = tf.nn.conv2d(X, W1, strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.relu(L1)
# Pooling 역시 tf.nn.max_pool 을 이용하여 쉽게 구성할 수 있습니다.
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
# L1 = tf.nn.dropout(L1, keep_prob)

# L2 Conv shape=(?, 33, 33, 36)
#    Pool     ->(?, 17, 17, 36)
W2 = tf.Variable(tf.random_normal([5, 5, 24, 36], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
# L2 = tf.nn.dropout(L2, keep_prob)

# L3 Conv shape=(?, 17, 17, 48)
#    Pool     ->(?, 9, 9, 48)
W3 = tf.Variable(tf.random_normal([5, 5, 36, 48], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 2, 2, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

# L4 Conv shape=(?, 17, 17, 48)
#    Pool     ->(?, 9, 9, 48)
W4 = tf.Variable(tf.random_normal([3, 3, 48, 64], stddev=0.01))
L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')


model.add(Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (1, 1)))
model.add(Dropout(0.5))

# L5 Conv shape=(?, 17, 17, 48)
#    Pool     ->(?, 9, 9, 48)
W4 = tf.Variable(tf.random_normal([3, 3, 48, 64], stddev=0.01))
L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')



# L2 = tf.nn.dropout(L2, keep_prob)
# FC 레이어: 입력값 7x7x64 -> 출력값 256
# Full connect를 위해 직전의 Pool 사이즈인 (?, 7, 7, 64) 를 참고하여 차원을 줄여줍니다.
#    Reshape  ->(?, 256)
W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))
L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob)

# 최종 출력값 L3 에서의 출력 256개를 입력값으로 받아서 0~9 레이블인 10개의 출력값을 만듭니다.
W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L3, W4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
# 최적화 함수를 RMSPropOptimizer 로 바꿔서 결과를 확인해봅시다.
# optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)






# origin network model
model = Sequential()
# model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape = (myconfig.resize_image_height, myconfig.resize_image_width, 3)))
# model.add(Lambda(input_shape = (myconfig.resize_image_height, myconfig.resize_image_width, 3)))

# convolution layer : input image size
# model.add(Conv2D(24, (5, 5), padding = 'same', strides = (2, 2), activation = "relu", input_shape = (myconfig.resize_image_height, myconfig.resize_image_width, 3)))
# model.add(MaxPooling2D(pool_size = (2, 2), strides = (1, 1)))
# model.add(Dropout(0.25))
# model.add(Conv2D(36, (5, 5), padding = 'same', strides = (2, 2), activation = "relu"))
# model.add(MaxPooling2D(pool_size = (2, 2), strides = (1, 1)))
# model.add(Dropout(0.25))
# model.add(Conv2D(48, (5, 5), padding = 'same', strides = (2, 2), activation = "relu"))
# model.add(MaxPooling2D(pool_size = (2, 2), strides = (1, 1)))
# model.add(Dropout(0.25))
# model.add(Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', activation = "relu"))
# model.add(MaxPooling2D(pool_size = (2, 2), strides = (1, 1)))
# model.add(Dropout(0.25))
# model.add(Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', activation = "relu"))
# model.add(MaxPooling2D(pool_size = (2, 2), strides = (1, 1)))
# model.add(Dropout(0.5))
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


# model = Sequential()
# model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape = (myconfig.resize_image_height, myconfig.resize_image_width, 3)))
#
# # convolution layer : input image size
# model.add(Conv2D(24, (7, 7), padding = 'same', strides = (2, 2), activation = "relu"))
# model.add(MaxPooling2D(pool_size = (2, 2), strides = (1, 1)))
# model.add(Dropout(0.25))
# model.add(Conv2D(24, (7, 7), padding = 'same', strides = (2, 2), activation = "relu"))
# model.add(MaxPooling2D(pool_size = (2, 2), strides = (1, 1)))
# model.add(Dropout(0.25))
# model.add(Conv2D(24, (5, 5), padding = 'same', strides = (2, 2), activation = "relu"))
# model.add(MaxPooling2D(pool_size = (2, 2), strides = (1, 1)))
# model.add(Dropout(0.25))
# model.add(Conv2D(36, (5, 5), padding = 'same', strides = (2, 2), activation = "relu"))
# model.add(MaxPooling2D(pool_size = (2, 2), strides = (1, 1)))
# model.add(Dropout(0.25))
# model.add(Conv2D(48, (5, 5), padding = 'same', strides = (2, 2), activation = "relu"))
# model.add(MaxPooling2D(pool_size = (2, 2), strides = (1, 1)))
# model.add(Dropout(0.25))
# model.add(Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', activation = "relu"))
# model.add(MaxPooling2D(pool_size = (2, 2), strides = (1, 1)))
# model.add(Dropout(0.25))
# model.add(Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', activation = "relu"))
# model.add(MaxPooling2D(pool_size = (2, 2), strides = (1, 1)))
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(1164))
# model.add(Activation('relu'))
# model.add(Dense(100))
# model.add(Activation('relu'))
# model.add(Dense(50))
# model.add(Activation('relu'))
# model.add(Dense(10))
# model.add(Activation('relu'))
# model.add(Dense(1))
# model.summary()
# model.compile(loss = 'mse', optimizer = Adam(1e-4))


local_epoch = [6]
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

    history = model.fit_generator(train_generator, steps_per_epoch = myconfig.steps_per_epoch, validation_data = validation_generator, validation_steps = myconfig.validation_steps, epochs = local_epoch[i], verbose = 1)

    print(history.history['loss'])
    print(history.history['val_loss'])

    modelname = "model_%s-2_%s_%s_%s_full.h5" % (today, myconfig.resize_image_width, myconfig.resize_image_height, local_epoch[i])
    # modelname = "model.h5"
    model.save(modelname)
    # model.save(myconfig.modelname)
    endtime = datetime.now()
    print("end time : ", endtime-starttime)

    log = "%s, %s, %s\n" % (history.history['loss'], history.history['val_loss'], (endtime-starttime))
    fhistory.write(log)
    fhistory = open(filename, 'a')
