---
layout:     post
title:      Tensorflow
subtitle:   keras模型实现断点续训
date:       2020-03-23
author:     张宇伟
header-img: img/blog/desk_blue.jpg
catalog: true
tags:
    - python
    - Machine Learning
    - Tensorflow

---

实现断开模型后还能接着继续训练。这里利用ModelCheckpoint实现保存模型。

```python
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json
from keras.models import load_model
```

先搭建一个模型框架
```python
nn_model = Sequential()
nn_model.add(Dense(32, input_dim=4, activation='relu'))
nn_model.add(Dense(16, input_dim=32, activation='relu'))
nn_model.add(Dense(8, input_dim=16, activation='relu'))
nn_model.add(Dense(1))
nn_model.compile(loss='mean_squared_error', optimizer='adam')
```
模型框架搭建完成，我这里添加了一个early_stop(告诉模型什么时候停止，防止没有必要的训练)和**ModelCheckpoint(告诉模型什么情况下保存模型，以便于以后继续训练)。这里我们想要保存val_loss最小时的模型。**意思即是val_loss每个epoch不断下降，模型就会被保存。
**注意：ModelCheckpoint的一个参数save_weights_only默认为False, 意思就是保存整个模型，包括模型的结构、权重、训练配置（损失函数、优化器等）、优化器的状态。**
```python
early_stop = EarlyStopping(monitor='val_loss', patience=1000, verbose=1)
checkpoint_save_dir = 'D:\\PycharmProjects\\dnn\\check_points'
if not os.path.isdir(checkpoint_save_dir):
    os.makedirs(checkpoint_save_dir)
# checkpoint = ModelCheckpoint(os.path.join(checkpoint_save_dir, "model_{epoch:02d}-{val_loss:.2f}.hdf5"),
#                              monitor='val_loss', verbose=1, save_best_only=True, mode='min')
checkpoint = ModelCheckpoint(os.path.join(checkpoint_save_dir, "model.hdf5"),
                             monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint, early_stop]
history = nn_model.fit(X_train_sc, y_train, epochs=10000, batch_size=1, verbose=1,
                           callbacks=callbacks_list, validation_split=0.3,
                           shuffle=False)
```
**如果保存的是整个模型**，ModelCheckpoint的参数save_weights_only=False保持默认, 加载训练很简单, 直接load_model即可。
**如果只保存了权重**，那么你需要重新指明之前模型框架，损失函数，优化器。
具体代码如下
```python
if save_weights_only:
	nn_model = Sequential()
	nn_model.add(Dense(32, input_dim=4, activation='relu'))
	nn_model.add(Dense(16, input_dim=32, activation='relu'))
	nn_model.add(Dense(8, input_dim=16, activation='relu'))
	nn_model.add(Dense(1))
    nn_model.load_weights(checkpoint_directory_path)
    nn_model.compile(loss='mean_squared_error', optimizer='adam')
else:
    nn_model = load_model(checkpoint_directory_path)
history = nn_model.fit(X_train_sc, y_train, epochs=10000, batch_size=1, verbose=1,
                           callbacks=callbacks_list, validation_split=0.3,
                           shuffle=False)
```