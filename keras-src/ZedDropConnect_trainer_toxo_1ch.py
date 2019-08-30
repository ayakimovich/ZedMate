import numpy as np
import os
import time
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models, optimizers, callbacks
from keras import backend as K
from keras.utils import to_categorical
from utils import plot_log
from skimage.transform import resize
from sklearn.utils import class_weight
from utils import report, h5_read, load_mnist
from ddrop_layers import DropConnectDense, DropConnect



K.set_image_data_format('channels_last')

def DropConnectNet(input_shape, n_class):

    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(128, activation='relu')(inputs)
    x = DropConnect(layers.Dense(64, activation='relu'), prob=0.5)(x)
    predictions = layers.Dense(n_class, activation='softmax')(x)
    
    model = models.Model(input=inputs, output=predictions)
    return model

def train(save_dir, batch_size, lr, shift_fraction, epochs, model, data, running_time):
    
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data
    class_weights_array = class_weight.compute_class_weight(
                                                            #None
                                                            'balanced'
                                               ,np.unique(np.argmax(y_train, axis=1))
                                               ,np.argmax(y_train, axis=1))
    
    class_weights={0:class_weights_array[1],1:class_weights_array[0]}
    # callbacks
    
    log_dir = save_dir + '\\tensorboard-logs-dd' + '\\' + running_time
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log = callbacks.CSVLogger(save_dir + '\\log-dd.csv')
    tb = callbacks.TensorBoard(log_dir=log_dir,
                               batch_size=batch_size)
    checkpoint = callbacks.ModelCheckpoint(save_dir + '\\weights-dd-{epoch:02d}.h5', monitor='val_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Begin: Training with data augmentation---------------------------------------------------------------------#
    
    def train_generator(x, y, batch_size, savedir, shift_fraction=0.):
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           samplewise_std_normalization = False,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size, shuffle=False)
        while 1:
            x_batch, y_batch = generator.next()
            yield (x_batch, y_batch)

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    print(class_weights)

    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              callbacks=[log, tb, checkpoint],
              class_weight= class_weights,
              shuffle=True)

    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(save_dir + 'trained_model_dd_toxo.h5')
    print('Trained model saved to \'%s \\trained_mode_dd_toxo.h5\'' % save_dir)

    
    plot_log(os.path.join(save_dir, 'log-dd.csv'), show=True)

    return model


running_time =time.strftime('%b-%d-%Y_%H-%M')
epochs=300
batch_size=100 # 5000
lr=0.001 # Initial learning rate 0.001 0.000001

shift_fraction=0.0 # Fraction of pixels to shift at most in each direction


test_data_size = False

load_dir=os.path.join('../tensors','toxo')

save_dir=os.path.join(load_dir,'training_toxo_dropconnect',running_time)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# load data
x_train, h5file = h5_read(os.path.join(load_dir,"data_train-bn-wl-1chV4_balanced.hdf5"))
h5file.close()
y_train, h5file = h5_read(os.path.join(load_dir,"labels_train-bn-wl-1chV4_balanced.hdf5"))
h5file.close()
y_train = y_train.flatten()

x_validate, h5file = h5_read(os.path.join(load_dir,"data_validate-bn-wl-1chV4_balanced.hdf5"))
h5file.close()
y_validate, h5file = h5_read(os.path.join(load_dir,"labels_validate-bn-wl-1chV4_balanced.hdf5"))
h5file.close()
y_validate = y_validate.flatten()


#mimicry_embed

x_train = resize(x_train, (20, 8, ), anti_aliasing=True)
x_validate = resize(x_validate, (20, 8, ), anti_aliasing=True)

x_train = np.pad(x_train, [(4, 4), (10, 10), (0, 0)], mode='constant', constant_values=0)
x_validate = np.pad(x_validate, [(4, 4), (10, 10), (0, 0)], mode='constant', constant_values=0)



x_train = np.swapaxes(x_train,0,2)
x_validate = np.swapaxes(x_validate,0,2)


y_train = to_categorical(y_train.astype('float32'))
y_validate = to_categorical(y_validate.astype('float32'))

#substitute by mnist for initialization
#(x_train, y_train), (x_validate, y_validate) = load_mnist()



# reshaping for dropconnect
x_train = x_train.reshape(x_train.shape[0], -1)
x_validate = x_validate.reshape(x_validate.shape[0], -1)

model = DropConnectNet(input_shape=x_train.shape[1:], n_class=len(np.unique(np.argmax(y_train, 1))))

model.summary()

model.load_weights('../models/toxo_dropconnect-1chan.h5')


#train(save_dir, batch_size, lr, shift_fraction, epochs, data=((x_train, y_train), (x_validate, y_validate)), model=model, running_time=running_time)


report(model,x_validate,y_validate, os.path.join(save_dir,'report.txt'))