## Import Module


```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, SeparableConv2D, AvgPool2D, MaxPool2D, GlobalAvgPool2D
from tensorflow.keras.layers import Flatten, Activation, Dropout, BatchNormalization, Input, Add, GlobalAvgPool2D, Reshape, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
```

## Layer Functions


```python
def conv2d_bn(x, filters, kernel_size, padding='same', strides=1, activation='elu', weight_decay=1e-5, bn = True):
    x = Conv2D(filters, kernel_size, padding=padding, strides=strides, kernel_regularizer=l2(weight_decay))(x)
    if bn: x = BatchNormalization()(x)
    if activation: x = Activation(activation)(x)
    return x

def sepconv2d_bn(x, filters, kernel_size, padding='same', strides=1, activation='elu', weight_decay=1e-5, depth_multiplier=1, bn = True):
    x = SeparableConv2D(filters, kernel_size, padding=padding, strides=strides, depth_multiplier=depth_multiplier, depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay))(x)
    if bn: x = BatchNormalization()(x)
    if activation: x = Activation(activation)(x)
    return x
```

## LR Scheduler


```python
def lr_scheduler(epoch, lr):
  if epoch < 5 or epoch > 60:
    print('LR : ', lr)
    return lr
  else:
    print('LR : ', lr*0.95)
    return lr * 0.95

callback1 = LearningRateScheduler(lr_scheduler)
```

## Load mnist data


```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
```

    (60000, 28, 28)
    (10000, 28, 28)
    (60000,)
    (10000,)
    

## Define model


```python
model_input = Input(shape=(28,28))
x = model_input
x = Reshape((28,28,1))(x)

res = conv2d_bn(x, 64, 1, strides=2, activation=None)
x = conv2d_bn(x, 64, 3, activation='elu')
x = conv2d_bn(x, 64, 3, activation=None)
x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
x = Add()([x, res])
x = Activation('elu')(x)
x = Dropout(0.25)(x)

res = conv2d_bn(x, 128, 1, strides=2, activation=None)
x = conv2d_bn(x, 128, 3, activation='elu')
x = conv2d_bn(x, 128, 3, activation=None)
x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
x = Add()([x, res])
x = Activation('elu')(x)
x = Dropout(0.25)(x)

res = conv2d_bn(x, 256, 1, strides=2, activation=None)
x = conv2d_bn(x, 256, 3, activation='elu')
x = conv2d_bn(x, 256, 3, activation='elu')
x = conv2d_bn(x, 256, 3, activation=None)
x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
x = Add()([x, res])
x = Activation('elu')(x)
x = Dropout(0.25)(x)

res = conv2d_bn(x, 512, 1, strides=2, activation=None)
x = conv2d_bn(x, 512, 3, activation='elu')
x = conv2d_bn(x, 512, 3, activation='elu')
x = conv2d_bn(x, 512, 3, activation=None)
x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
x = Add()([x, res])
x = Activation('elu')(x)
x = Dropout(0.25)(x)

x = GlobalAvgPool2D()(x)

model_output = Dense(10, activation="softmax")(x)

model = Model(model_input, model_output)

optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True)
model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])
# model.summary()
```

## Train model


```python
model.fit(
    x_train, y_train, 
    validation_data=(x_test, y_test), 
    epochs=30, batch_size=128, callbacks=[callback1])
```

    Epoch 1/30
    LR :  0.0010000000474974513
    469/469 [==============================] - 26s 51ms/step - loss: 0.9197 - acc: 0.7185 - val_loss: 0.9748 - val_acc: 0.7275
    Epoch 2/30
    LR :  0.0010000000474974513
    469/469 [==============================] - 23s 50ms/step - loss: 0.1888 - acc: 0.9500 - val_loss: 0.1193 - val_acc: 0.9715
    Epoch 3/30
    LR :  0.0010000000474974513
    469/469 [==============================] - 23s 50ms/step - loss: 0.1417 - acc: 0.9659 - val_loss: 0.0876 - val_acc: 0.9795
    Epoch 4/30
    LR :  0.0010000000474974513
    469/469 [==============================] - 23s 49ms/step - loss: 0.1198 - acc: 0.9708 - val_loss: 0.0704 - val_acc: 0.9858
    Epoch 5/30
    LR :  0.0010000000474974513
    469/469 [==============================] - 23s 50ms/step - loss: 0.1049 - acc: 0.9766 - val_loss: 0.0709 - val_acc: 0.9851
    Epoch 6/30
    LR :  0.0009500000451225787
    469/469 [==============================] - 23s 50ms/step - loss: 0.0960 - acc: 0.9798 - val_loss: 0.0738 - val_acc: 0.9850
    Epoch 7/30
    LR :  0.0009025000152178108
    469/469 [==============================] - 23s 50ms/step - loss: 0.0897 - acc: 0.9818 - val_loss: 0.0629 - val_acc: 0.9884
    Epoch 8/30
    LR :  0.0008573750033974647
    469/469 [==============================] - 23s 50ms/step - loss: 0.0844 - acc: 0.9826 - val_loss: 0.0640 - val_acc: 0.9885
    Epoch 9/30
    LR :  0.0008145062311086804
    469/469 [==============================] - 23s 49ms/step - loss: 0.0818 - acc: 0.9838 - val_loss: 0.0558 - val_acc: 0.9907
    Epoch 10/30
    LR :  0.0007737808919046074
    469/469 [==============================] - 23s 50ms/step - loss: 0.0767 - acc: 0.9854 - val_loss: 0.0601 - val_acc: 0.9900
    Epoch 11/30
    LR :  0.000735091819660738
    469/469 [==============================] - 23s 49ms/step - loss: 0.0787 - acc: 0.9849 - val_loss: 0.0570 - val_acc: 0.9907
    Epoch 12/30
    LR :  0.0006983372120885178
    469/469 [==============================] - 23s 50ms/step - loss: 0.0754 - acc: 0.9857 - val_loss: 0.0607 - val_acc: 0.9899
    Epoch 13/30
    LR :  0.0006634203542489559
    469/469 [==============================] - 23s 50ms/step - loss: 0.0701 - acc: 0.9864 - val_loss: 0.0574 - val_acc: 0.9915
    Epoch 14/30
    LR :  0.0006302493420662358
    469/469 [==============================] - 23s 50ms/step - loss: 0.0714 - acc: 0.9870 - val_loss: 0.0570 - val_acc: 0.9911
    Epoch 15/30
    LR :  0.0005987368611386045
    469/469 [==============================] - 23s 50ms/step - loss: 0.0669 - acc: 0.9883 - val_loss: 0.0638 - val_acc: 0.9895
    Epoch 16/30
    LR :  0.0005688000208465382
    469/469 [==============================] - 23s 50ms/step - loss: 0.0668 - acc: 0.9878 - val_loss: 0.0568 - val_acc: 0.9918
    Epoch 17/30
    LR :  0.0005403600225690752
    469/469 [==============================] - 23s 50ms/step - loss: 0.0691 - acc: 0.9880 - val_loss: 0.0528 - val_acc: 0.9926
    Epoch 18/30
    LR :  0.0005133419937919825
    469/469 [==============================] - 23s 50ms/step - loss: 0.0642 - acc: 0.9895 - val_loss: 0.0525 - val_acc: 0.9924
    Epoch 19/30
    LR :  0.0004876748775132
    469/469 [==============================] - 23s 50ms/step - loss: 0.0632 - acc: 0.9900 - val_loss: 0.0544 - val_acc: 0.9911
    Epoch 20/30
    LR :  0.00046329112810781223
    469/469 [==============================] - 23s 50ms/step - loss: 0.0648 - acc: 0.9889 - val_loss: 0.0524 - val_acc: 0.9923
    Epoch 21/30
    LR :  0.00044012657308485355
    469/469 [==============================] - 23s 50ms/step - loss: 0.0606 - acc: 0.9898 - val_loss: 0.0532 - val_acc: 0.9923
    Epoch 22/30
    LR :  0.00041812024719547477
    469/469 [==============================] - 23s 50ms/step - loss: 0.0602 - acc: 0.9902 - val_loss: 0.0515 - val_acc: 0.9931
    Epoch 23/30
    LR :  0.00039721422654110934
    469/469 [==============================] - 23s 50ms/step - loss: 0.0639 - acc: 0.9893 - val_loss: 0.0557 - val_acc: 0.9922
    Epoch 24/30
    LR :  0.00037735351797891776
    469/469 [==============================] - 23s 50ms/step - loss: 0.0589 - acc: 0.9899 - val_loss: 0.0562 - val_acc: 0.9923
    Epoch 25/30
    LR :  0.00035848583793267607
    469/469 [==============================] - 23s 50ms/step - loss: 0.0599 - acc: 0.9901 - val_loss: 0.0504 - val_acc: 0.9934
    Epoch 26/30
    LR :  0.00034056155709549785
    469/469 [==============================] - 23s 50ms/step - loss: 0.0576 - acc: 0.9912 - val_loss: 0.0521 - val_acc: 0.9933
    Epoch 27/30
    LR :  0.00032353347924072293
    469/469 [==============================] - 23s 50ms/step - loss: 0.0568 - acc: 0.9908 - val_loss: 0.0516 - val_acc: 0.9932
    Epoch 28/30
    LR :  0.00030735681357327847
    469/469 [==============================] - 23s 49ms/step - loss: 0.0572 - acc: 0.9911 - val_loss: 0.0502 - val_acc: 0.9934
    Epoch 29/30
    LR :  0.00029198898118920624
    469/469 [==============================] - 23s 50ms/step - loss: 0.0591 - acc: 0.9900 - val_loss: 0.0524 - val_acc: 0.9926
    Epoch 30/30
    LR :  0.00027738953212974593
    469/469 [==============================] - 23s 49ms/step - loss: 0.0575 - acc: 0.9916 - val_loss: 0.0514 - val_acc: 0.9930
    




    <tensorflow.python.keras.callbacks.History at 0x7fea0609ee10>



## Show model configuration


```python
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

%matplotlib inline

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
```




    
![svg](output_13_0.svg)
    




```python

```
