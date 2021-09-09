import numpy as np
from keras.optimizers import RMSprop
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras import models,layers,regularizers

from matplotlib  import pyplot as plt
train_page = np.loadtxt('mnist_train.csv' , delimiter="," , dtype='float', skiprows=1)
test_page = np.loadtxt('mnist_test.csv' , delimiter="," , dtype='float', skiprows=1)
train_im = train_page[:,1:]
test_im = test_page[:,1:]
train_label = to_categorical(train_page[:,0])
test_label = to_categorical(test_page[:,0])

print(train_im.shape)
print(train_label.shape)
print(train_page[:,0].shape)


network = models.Sequential()
network.add(layers.Dense(units=200 , activation='relu' , input_shape=(28*28, ) ,
                         kernel_regularizer=regularizers.l1(0.0002) ))
network.add(layers.Dropout(0.05))
network.add(layers.Dense(units=100, activation='relu',
                         kernel_regularizer=regularizers.l1(0.0001)))
network.add(layers.Dropout(0.05))
network.add(layers.Dense(units=10 , activation='softmax'))

#编译
network.compile(optimizer = RMSprop(lr = 0.001) , loss='categorical_crossentropy' , metrics=['accuracy'])

# 训练网络 epochs表示训练多少个回合， batch_size表示每次训练给多大的数据
history = network.fit(train_im, train_label, epochs=25, batch_size=128, verbose=2)
print(history)

plt.plot(np.arange(len(history.history['loss'])),history.history['loss'],label='training')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc=0)
plt.show()


test_loss , test_accuracy = network.evaluate(test_im,test_label)
print("test_loss:", test_loss, "    test_accuracy:", test_accuracy)

network.save('./MIN_model.h5')

