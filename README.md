# implementation_FL

## Instrucciones
### Ejecutar servidor
- Definir el modelo, se debe cambiar según se requiera, por defecto será CNN 1D (comentar el que no se utilice). Considerar que se debe usar el mismo modelo que usa el cliente.
```
    #modelo cnn1D
    model = create_CNN1D_model(n_timesteps, n_features, n_outputs)

    #modelo DNN
    #model = create_DNN_model(n_timesteps,n_features,n_outputs)
```
- Al definir una estrategia se debe indicar el número mínimo de clientes usados durante el entrenamiento (por ronda), número mínimo de clientes en el sistema para que se comience el entrenamiento, por defecto estos parámetros son 10 los dos.
```
    #Se define una estrategia
    strategy = fl.server.strategy.FedAvg(
      eval_fn= get_eval_fn(model,x_val, y_val , accuracy_list), #¡evaluation
      fraction_fit=0.1,  # Muestra el 10% de los clientes disponibles para la siguiente ronda
      min_fit_clients=10,  # el número mínimo de clientes usados durante el entrenamiento (por ronda)
      min_available_clients=10,  # número mínimo de clientes en el sistema para que se comience el entrenamiento
    )
```
- Por último se debe indicar el puerto (```0.0.0.0:7000```, por defecto), el cual tiene que ser el mismo que el de los clientes, y el número de rondas que se requieran, por defecto estas son 20.
```
    fl.server.start_server("0.0.0.0:7000", config={"num_rounds": 20}, strategy=strategy)
```
- Ahora que el server está listo, ya se puede ejecutar: ```python server.py```, esperar unos segundos y ya se puede ejecutar algunos clientes.
### Ejecutar cliente
- Definir el modelo, se debe cambiar según se requiera, por defecto será CNN 1D (comentar el que no se utilice). Considerar que se debe usar el mismo modelo que usa el servidor.
```
    #modelo cnn1D
    model = create_CNN1D_model(n_timesteps, n_features, n_outputs)

    #modelo DNN
    #model = create_DNN_model(n_timesteps,n_features,n_outputs)
```
- En la función fit se pueden definir el número de épocas que se requiere que haga el cliente en el entrenamiento local, por defecto este parametro es 5.
```
    def fit(self, parameters, config):
      model.set_weights(parameters)
      model.fit(x_train, y_train, epochs=5, batch_size=32) 
      return model.get_weights(), len(x_train), {}
```
- Por último se debe indicar el puerto (```0.0.0.0:7000```, por defecto), el cual tiene que ser el mismo que el del servidor.
```
   fl.client.start_numpy_client(server_address="0.0.0.0:7000", client=CifarClient())
```
 - Para ejecutar el cliente, se debe tener en cuenta los datos que se le quieran pasar desde el conjunto de datos, por ejemplo el cliente 1 tendrá los datos de 2 sujetos de entrenamiento del conjunto de datos, del sujeto 1 y del sujeto 3, Entonces se ejectua de la siguiente manera: ```python client.py 1 3```.
En cambio el cliente 10, tiene los datos de 3 sujetos del conjunto de datos, el sujeto 28, 29 y 30, por lo que se ejecuta de la siguiente manera ```python client.py 28 30```. El conjunto de datos que se le asignen a cada cliente, es un rango entre el primer dato del primer sujeto y el último dato del segundo sujeto. Sólo pueden ser utilizados los sujetos del grupo de entrenamiento del dataset, los cuales son 1,  3,  5,  6,  7,  8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29 y 30.

- Para mayor simplicidad ejecutar el archivo ```run_10_clients.sh```, con ```bash run_10_clients.sh```, el cual ejecuta 10 clientes (donde cada cliente tiene los datos de 2 sujetos del conjunto de datos de entrenamiento).

### Referencias
- Dataset: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
- Preproceso: https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
