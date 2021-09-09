import numpy as np
import tensorflow as tf
import pandas as pd
test_page = np.loadtxt('test.csv' , delimiter="," , dtype='float', skiprows=1)
model = tf.keras.models.load_model('./MIN_model.h5')
prediction = model.predict(test_page[:])
result = np.argmax(prediction,axis = 1)#选取概率最大的数值
a = []
for i in range(1,28001):
    a.append(i)
dataframe = pd.DataFrame({'ImageId':a,'Label':result})
dataframe.to_csv("rre.csv",index=False,sep=',')