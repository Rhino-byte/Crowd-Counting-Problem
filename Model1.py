from keras import Input, Model, Sequential
#from keras.layers import Conv2D, MaxPooling2D, Concatenate, Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import Conv2D,Dense,concatenate,Activation,Dropout,Input

#Input layer
input_layer = Input(shape=input_shape)

#Convolution Parallel layer 1
Conv_Layer_1 = tf.keras.layers.Conv2D(10, kernel_size=(9, 9), padding='same',activation='relu', input_shape=input_shape)(input_layer)
inputlayer_1_pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(Conv_Layer_1)
inputlayer_1_pooling1= tf.keras.layers.Dropout(0.25)(inputlayer_1_pooling)

#Convolution Parallel layer 2
Conv_Layer_2 = tf.keras.layers.Conv2D(14, kernel_size=(7, 7),padding='same',activation='relu', input_shape=input_shape)(input_layer)
inputlayer_2_pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(Conv_Layer_2)
inputlayer_2_pooling1= tf.keras.layers.Dropout(0.25)(inputlayer_2_pooling)

#Convolution Parallel layer 3
Conv_Layer_3 = tf.keras.layers.Conv2D(16, kernel_size=(5, 5),padding='same',activation='relu', input_shape=input_shape)(input_layer)
inputlayer_3_pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(Conv_Layer_3)
inputlayer_3_pooling1= tf.keras.layers.Dropout(0.25)(inputlayer_3_pooling)

#Perform concantenation of all three parallel layers
concatenation_layer = concatenate(inputs=[inputlayer_1_pooling1
                                   ,inputlayer_2_pooling1,inputlayer_3_pooling1
                                  ],name="concat")

#Add 6 convolutional layers as per above architecture

convlayer2 = tf.keras.layers.Conv2D(40, kernel_size=(3,3), padding='same',activation='relu', input_shape=input_shape)(concatenation_layer)
convlayer2_1= tf.keras.layers.BatchNormalization()(convlayer2)

convlayer3 = tf.keras.layers.Conv2D(60, kernel_size=(3,3), padding='same',activation='relu', input_shape=input_shape)(convlayer2_1)
polling_layer3_1= tf.keras.layers.Dropout(0.25)(convlayer3)
convlayer3_1= tf.keras.layers.BatchNormalization()(polling_layer3_1)
polling_layer3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(convlayer3_1)


convlayer4 = tf.keras.layers.Conv2D(40, kernel_size=(3,3), padding='same',activation='relu', input_shape=input_shape)(polling_layer3)
polling_layer4_1= tf.keras.layers.Dropout(0.25)(convlayer4)
convlayer4_1= tf.keras.layers.BatchNormalization()(polling_layer4_1)
polling_layer4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(convlayer4_1)


convlayer5 = tf.keras.layers.Conv2D(20, kernel_size=(3,3), padding='same',activation='relu', input_shape=input_shape)(polling_layer4)
polling_layer5_1= tf.keras.layers.Dropout(0.25)(convlayer5)
convlayer5_1= tf.keras.layers.BatchNormalization()(polling_layer5_1)


convlayer6 = tf.keras.layers.Conv2D(10, kernel_size=(3,3), padding='same',activation='relu', input_shape=input_shape)(convlayer5_1)
polling_layer6_1= tf.keras.layers.Dropout(0.25)(convlayer6)
convlayer6_1= tf.keras.layers.BatchNormalization()(polling_layer6_1)

flatten = tf.keras.layers.Flatten()(convlayer6_1)
#denss1 = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)(flatten)
output = tf.keras.layers.Dense(1)(flatten)


from tensorflow.keras.models import Model

final_model = Model(inputs=input_layer,outputs=output)
