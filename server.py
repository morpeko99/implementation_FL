from models import create_CNN1D_model, create_DNN_model
from preproc import load_dataset_server

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import time

import flwr as fl

# Start Flower server for three rounds of federated learning
def main() -> None:
    start = time.time()
    #Se carga la data para evaluar el modelo global en cada ronda (data que es de clientes del conjunto de test)
    (x_val, y_val)= load_dataset_server()
    
     #se obtienen las dimensiones de los pasos de tiempo (128), cantidad de caracteristicas (9) y numero de clases (6)
    n_timesteps, n_features, n_outputs = x_val.shape[1], x_val.shape[2], y_val.shape[1]
    
    #Se selcciona el modelo que se quiera utilizar (debe ser el mismo que use el cliente):

    #modelo cnn1D
    model = create_CNN1D_model(n_timesteps, n_features, n_outputs)

    #modelo DNN
    #model = create_DNN_model(n_timesteps,n_features,n_outputs)
    
    #para describir la red neuronal:
    #model.summary()

    #lista de accuracy por ronda
    accuracy_list = []
    
    #Se define una estrategia
    strategy = fl.server.strategy.FedAvg(
        eval_fn= get_eval_fn(model,x_val, y_val , accuracy_list), #¡evaluation
        fraction_fit=0.1,  # Muestra el 10% de los clientes disponibles para la siguiente ronda
        min_fit_clients=10,  # el número mínimo de clientes usados durante el entrenamiento (por ronda)
        min_available_clients=10,  # número mínimo de clientes en el sistema para que se comience el entrenamiento
        )
    
    fl.server.start_server("0.0.0.0:7000", config={"num_rounds": 20}, strategy=strategy)
    np.savetxt('accuracy_per_round.csv', accuracy_list) #se guardan las accuracy obtenidas por cada ronda
    end = time.time()
    print("Tiempo de ejecución: ", (end-start))
def get_eval_fn(model, x_val, y_val, accuracy_list):
    """Retorna una función de evaluación para la evaluación del lado del servidor"""


    """función de evaluación que será llamada en cada ronda"""
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:

         # se actualizan los pesos al modelo
        model.set_weights(weights) 
        loss, accuracy = model.evaluate(x_val, y_val, batch_size=32, verbose=0)

        #se guarda el accuracy de la ronda en una lista
        accuracy_list.append(accuracy)

        return loss, {"accuracy": accuracy}


    return evaluate


if __name__ == "__main__":
    
    main()