from capsulelayers import PrimaryCap, CapsuleLayer, Length, Mask, margin_loss, test, generateRepresentativeImages
import numpy as np
import os
from sklearn.utils import class_weight
from utils import combine_images, report, h5_read, load_mnist, mimicry_embed, plot_log
from keras import layers, models, optimizers, callbacks
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import time


K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)
     # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
    #digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=8, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model

def train(save_dir, batch_size, lr, lr_decay_value, lam_recon, shift_fraction, epochs, model, data, running_time):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """

    
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data
    class_weights_array = class_weight.compute_class_weight('balanced'
                                               ,np.unique(np.argmax(y_train, axis=1))
                                               ,np.argmax(y_train, axis=1))
    
    class_weights={0:class_weights_array[1],1:class_weights_array[0]}
    # callbacks
    
    log_dir = save_dir + '/tensorboard-logs-linux' + '/' + running_time
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log = callbacks.CSVLogger(save_dir + '/log-linux.csv')
    tb = callbacks.TensorBoard(log_dir=log_dir,
                               batch_size=batch_size)
    checkpoint = callbacks.ModelCheckpoint(save_dir + '/weights-linux-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: lr * (lr_decay_value ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., lam_recon],
                  weighted_metrics={'capsnet': 'accuracy'},
                  metrics={'capsnet': 'accuracy'})

    """
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    """

    # Begin: Training with data augmentation---------------------------------------------------------------------#
    
    def train_generator(x, y, batch_size, savedir, shift_fraction=0.):
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           #brightness_range = (0.000009, 0.000001),
                                           samplewise_std_normalization = False,
                                           #rotation_range = 90,
                                           #zoom_range = [0.5 , 1],
                                           #horizontal_flip = True,
                                           #vertical_flip = True,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        #generator = train_datagen.flow(x, y, batch_size=batch_size)
        #generator = train_datagen.flow(x, y, batch_size=batch_size, shuffle=True,save_to_dir=savedir)
        generator = train_datagen.flow(x, y, batch_size=batch_size, shuffle=False)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    print(class_weights)
    model.fit_generator(generator=train_generator(x_train, y_train, batch_size, savedir=save_dir + '/data', shift_fraction=0.),
                        steps_per_epoch=int(y_train.shape[0] / batch_size),
                        epochs=epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, tb, checkpoint, lr_decay],
                        #nb_val_samples=200,
                        class_weight= class_weights,
                        shuffle=True)
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(save_dir + 'trained_model_toxo.h5')
    print('Trained model saved to \'%s \\trained_model_toxo.h5\'' % save_dir)

    
    plot_log(save_dir + '/log-linux.csv', show=True)

    return model

running_time =time.strftime('%b-%d-%Y_%H-%M')
epochs=200
batch_size=100 # 5000
lr=0.001 # Initial learning rate 0.001 0.000001
#lr_decay = 0.999 # The value multiplied by lr at each epoch. Set a larger value for larger epochs
lr_decay = 0.999 # The value multiplied by lr at each epoch. Set a larger value for larger epochs
lam_recon=0.392 # 0.392, 1.563
routings=3 # sNumber of iterations used in routing algorithm. should > 0
shift_fraction=0.0 # Fraction of pixels to shift at most in each direction
#debug', action='store_true', help="Save weights by TensorBoard")
#load_dir=os.path.join('R:\Dropbox (LMCB)\ZedMate_paper','tensors')

test_data_size = False

load_dir=os.path.join('../tensors','toxo')

save_dir=os.path.join(load_dir,'training_toxoV2',running_time)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# testing', action='store_true' Test the trained model on testing datase
# digit=5 # mnist Digit to manipulate
#('-w', '--weights', default=None, help="The path of the saved weights. Should be specified when testing")

# load data
x_train, h5file = h5_read(os.path.join(load_dir,"data_train-bn-wl-2chV4_balanced.hdf5"))
h5file.close()
y_train, h5file = h5_read(os.path.join(load_dir,"labels_train-bn-wl-2chV4_balanced.hdf5"))
h5file.close()
y_train = y_train.flatten()

x_validate, h5file = h5_read(os.path.join(load_dir,"data_validate-bn-wl-2chV4_balanced.hdf5"))
h5file.close()
y_validate, h5file = h5_read(os.path.join(load_dir,"labels_validate-bn-wl-2chV4_balanced.hdf5"))
h5file.close()
y_validate = y_validate.flatten()



x_train, x_validate = mimicry_embed(x_train, x_validate)

y_train = to_categorical(y_train.astype('float32'))
y_validate = to_categorical(y_validate.astype('float32'))

#substitute by mnist for initialization
#(x_train, y_train), (x_test, y_test) = load_mnist()

# define model
if test_data_size:
    samples_idx = np.random.randint(0, high=len(x_train)-1, size=10)
    x_train = x_train[samples_idx,:,:,:]
    y_train = y_train[samples_idx,:]


model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                 routings=routings)

model.summary()


model.load_weights('../models/toxo_2chan-2.h5')

#generateRepresentativeImages(eval_model, x_validate, y_validate, load_dir, 1, running_time)

#train(save_dir, batch_size, lr, lr_decay, lam_recon, shift_fraction, epochs, data=((x_train, y_train), (x_validate, y_validate)), model=model, running_time=running_time)


report(eval_model,x_validate,y_validate, os.path.join(save_dir,'report.txt'))
