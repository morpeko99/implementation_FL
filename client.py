from models import create_CNN1D_model, create_DNN_model
from preproc import load_dataset

import os
import sys as s
import numpy as np

import flwr as fl

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"


if __name__ == "__main__": 
    
    #se carga datos del cliente desde dataset HAR
    (x_train, y_train), (x_test, y_test) = load_dataset(client=(int(s.argv[1]),int(s.argv[2])))

     #se obtienen las dimensiones de los pasos de tiempo (128), cantidad de caracteristicas (9) y numero de clases (6)
    n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
    
    #Se selcciona el modelo que se quiera utilizar (debe ser el mismo que use el servidor):

    #modelo cnn1D
    model = create_CNN1D_model(n_timesteps,n_features,n_outputs)

    #modelo DNN
    #model = create_DNN_model(n_timesteps,n_features,n_outputs)

    # Se define un cliente Flower
    
    class CifarClient(fl.client.NumPyClient):
        
        def get_parameters(self):
            """Retorna los pesos del modelo local"""

            return model.get_weights()

        def fit(self, parameters, config):
            """se actualizan los pesos y se entrena el modelo para obtener un nuevo modelo local"""

            model.set_weights(parameters)
            model.fit(x_train, y_train, epochs=5, batch_size=32) 
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config,):
            """Se evalúa el modelo local"""

            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)                
            return loss, len(x_test), {"accuracy": accuracy}

    # Se inicia el cliente Flower
    fl.client.start_numpy_client(server_address="0.0.0.0:7000", client=CifarClient())