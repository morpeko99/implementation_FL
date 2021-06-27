# To store data
import pandas as pd
from numpy import dstack
from pandas import read_csv
# To do linear algebra
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical




def load_file(filepath, ini_idx, fin_idx):
    """Se carga un archivo como numpy array, solo las filas que indican los índices"""

    dataframe = read_csv(filepath, header=None, delim_whitespace=True,skiprows=ini_idx,nrows=(fin_idx-ini_idx+1))
    return dataframe.values

def load_group(filenames, ini_idx, last_idx, prefix=''):
    """Se carga una lista de archivos y se retorna como un numpy array 3d"""

    loaded = list()
    for name in filenames:
        data = load_file(prefix + name, ini_idx, last_idx)
        loaded.append(data)
    
    #se apila para tener las caracteristicas en tercera dimensión
    loaded = dstack(loaded)
    return loaded


def load_dataset_group(group, client, prefix=''):
    """Se cargan los datos de entrenamiento o de prueba"""

    #túpla que indica el índice del inicio y final de la data del cliente
    (ini_idx, last_idx) = client_index(group, client, prefix) 
    
    filepath = prefix + group + '/Inertial Signals/'
    
    #Se cargan los 9 archivos como una sola matriz 
    filenames = list()
    
    # total acceleration
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
    # body acceleration
    filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
    # body gyroscope
    filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
    
    #Se carga la data de salida
    X = load_group(filenames, ini_idx, last_idx, filepath)
    
    # Se carga las clases de salida
    y = load_file(prefix + group + '/y_'+group+'.txt', ini_idx, last_idx)
    
    return X, y
  


def load_dataset(prefix='', client=None):
    """carga el conjunto de datos, retorna los datos de prueba y entrenamiento para X e y """
    try:
        #se cargan los datos de cliente si sus datos están en el train
        X_client, y_client = load_dataset_group('train', client, prefix + './UCI_HAR_Dataset/')
        
        # Se disminuye en 1 para el one hot encode
        y_client = y_client - 1
        
        # one hot encode y
        y_client = to_categorical(y_client)
    except:
        #se cargan los datos del cliente si sus datos están en el test
        X_client, y_client = load_dataset_group('test', client, prefix + './UCI_HAR_Dataset/')
        
        # Se disminuye en 1 para el one hot encode
        y_client = y_client - 1
        
        # one hot encode y
        y_client = to_categorical(y_client)
        
    X_train, X_test, y_train, y_test = train_test_split(X_client, y_client, test_size = 0.3, random_state = 42)
    return (X_train, y_train), (X_test, y_test)

def client_index(group, client, prefix): #client debe ser un número que represente a una cliente del dataset

    ini, end = client
    df = pd.read_csv(prefix + group +'/subject_'+ group+'.txt', header=None)
    df_= df
    df1 = df.loc[(df[0] == ini)]
    df2 = df_.loc[(df[0] == end)]
    df = pd.concat([df1,df2])
    return (int(df.index[0]),int(df.index[-1])) #se retorna una tupla con el indice inicial e  indice final 


"""Las siguientes funciones son para cargar la data sin selección de data de clientes,
    ya que es para la evaluación del modelo global del lado del servidor, por lo que solo se necesita un conjunto de datos"""

def load_file_server(filepath):
    """Se carga un archivo como numpy array"""
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values

def load_group_server(filenames, prefix=''):
    """Se carga una lista de archivos y se retorna como un numpy array 3d"""

    loaded = list()
    for name in filenames:
        data = load_file_server(prefix + name)
        loaded.append(data)
    #se apila para tener las caracteristicas en tercera dimensión
    loaded = dstack(loaded)
    return loaded


def load_dataset_group_server(group, prefix=''):
    """Se cargan los datos de entrenamiento o de prueba"""
    
    filepath = prefix + group + '/Inertial Signals/'

    #Se cargan los 9 archivos como una sola matriz 
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']    
    # body acceleration
    filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']   
    # body gyroscope
    filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
    
    #Se carga la data de entrada
    X = load_group_server(filenames, filepath)
    # Se carga las clases de salida
    y = load_file_server(prefix + group + '/y_'+group+'.txt')
    
    return X, y

"""carga el conjunto de datos, retorna los datos de prueba y entrenamiento para X e y """
def load_dataset_server(prefix=''):
    #se cargan los datos de cliente si sus datos están en el train
    X_client, y_client = load_dataset_group_server('test', prefix + './UCI_HAR_Dataset/')
    print(X_client.shape, y_client.shape)
    
    #Se disminuye en 1 para el one hot encode
    y_client = y_client - 1
    
    # one hot encode y
    y_client = to_categorical(y_client)

    return X_client, y_client