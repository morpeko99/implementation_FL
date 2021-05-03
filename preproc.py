# To store data
import pandas as pd
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
# To do linear algebra
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical



'''def preproc():

    # Load dataset
    df = pd.read_csv("./train.csv")

    x = df.drop(['Activity'],axis=1)
    y = df['Activity']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

    return (X_train,y_train),(X_test,y_test)'''

# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
    # body acceleration
    filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
    # body gyroscope
    filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_'+group+'.txt')
    return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    #https://www.kaggle.com/srikanthreddyt/human-activity-recognition-using-conv1d
    # load all train
    trainX, trainy = load_dataset_group('train', prefix + './UCI_HAR_Dataset/')
    print(trainX.shape, trainy.shape)
    # load all test
    testX, testy = load_dataset_group('test', prefix + './UCI_HAR_Dataset/')
    print(testX.shape, testy.shape)
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return (trainX, trainy), (testX, testy)