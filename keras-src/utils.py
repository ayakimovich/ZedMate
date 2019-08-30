import numpy as np
from matplotlib import pyplot as plt
import csv
import math
import tables
from skimage.transform import resize
from sklearn.utils import class_weight
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc

  
def plot_log(filename, show=True):
    # load data
    keys = []
    values = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if keys == []:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                continue

            for _, value in row.items():
                values.append(float(value))

        values = np.reshape(values, newshape=(-1, len(keys)))
        values[:,0] += 1

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for i, key in enumerate(keys):
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for i, key in enumerate(keys):
        if key.find('acc') >= 0:  # acc
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()


def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

def h5_read(file):
    h5file = tables.open_file(file, driver="H5FD_CORE")
    array = h5file.root.somename.read()
    return array, h5file

def load_mnist():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # change number of classes to 2
    y_train[y_train >= 5] = 5
    y_train[y_train < 5] = 0
    y_train[y_train == 5] = 1
    y_test[y_test >= 5] = 5
    y_test[y_test < 5] = 0
    y_test[y_test == 5] = 1

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

def report (m,x,y_true,save_file):
    class_weights_array = class_weight.compute_class_weight('balanced'
                                               ,np.unique(np.argmax(y_true, axis=1))
                                               ,np.argmax(y_true, axis=1))
    class_weights=[class_weights_array[0],class_weights_array[1]]

    
    y_pred, x_recon = m.predict(x, batch_size=10)
    
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true.ravel(), y_pred.ravel())
    
    auc_keras = auc(fpr_keras, tpr_keras)
    y = np.argmax(y_pred,axis=1)
    
    y_true = np.argmax(y_true,axis=1)

    print('class weights: {},{}'.format(class_weights_array[0],class_weights_array[1]))
    print('precision score is: {}'.format(precision_score(y_true, y, average = 'weighted')))
    print('f1 score is: {}'.format(f1_score(y_true, y, average = 'weighted')))
    print('recall score is: {}'.format(recall_score(y_true, y, average = 'weighted')))
    print('confusion_matrix score is: {}'.format(confusion_matrix(y_true, y)))
    print('auc: {}'.format(auc_keras))
    #print('false positive rate - threshold: {} - {}'.format(fpr_keras, thresholds_keras))
    #print('true positive rate - threshold: {} - {}'.format(tpr_keras, thresholds_keras))
    l = 'class weights, {}:{} \n'.format(class_weights_array[0],class_weights_array[1])
    l = l +'precision score is, {} \n'.format(precision_score(y_true, y, average = 'weighted'))
    l = l +'f1 score is:, {} \n'.format(f1_score(y_true, y, average = 'weighted'))
    l = l +'recall score is, {} \n'.format(recall_score(y_true, y, average = 'weighted'))
    l = l +'confusion_matrix score is, {} \n'.format(confusion_matrix(y_true, y))
    l = l +'auc, {} \n'.format(auc_keras)
    l = l +'i,false_positive_rate \n'
    l = l + '\n'.join('{},{}'.format(*k) for k in enumerate(fpr_keras))
    l = l + '\ni,true_positive_rate \n'
    l = l + '\n'.join('{},{}'.format(*k) for k in enumerate(tpr_keras))
    l = l + 'i,threshold \n'
    l = l + '\n'.join('{},{}'.format(*k) for k in enumerate(thresholds_keras))
    
    f = open(save_file, "w")
    f.writelines(l)
    f.close()

def mimicry_embed(x_train, x_validate):
    #make the model same as mnist
    x_train = resize(x_train, (20, 8, ), anti_aliasing=True)
    x_validate = resize(x_validate, (20, 8, ), anti_aliasing=True)
    #x_train = np.pad(x_train, [(9, 9), (12, 12), (0, 0)], mode='constant', constant_values=0)
    #x_validate = np.pad(x_validate, [(9, 9), (12, 12), (0, 0)], mode='constant', constant_values=0)
    
    x_train = np.pad(x_train, [(4, 4), (10, 10), (0, 0)], mode='constant', constant_values=0)
    x_validate = np.pad(x_validate, [(4, 4), (10, 10), (0, 0)], mode='constant', constant_values=0)
    
    
    #x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
    #x_validate = x_validate.reshape(-1, 28, 28, 1).astype('float32')
    x_train = np.swapaxes(x_train,0,2)
    x_validate = np.swapaxes(x_validate,0,2)
    
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_validate = x_validate.reshape(-1, 28, 28, 1)
    return x_train, x_validate

    