# implementation_FL
### Índice
- [Ejecutar servidor](#id1)
- [Ejecutar cliente](#id2)
- [Salida](#id3)
## Instrucciones
### Ejecutar servidor<a name="id1"></a>
- Definir el modelo, se debe cambiar según se requiera, por defecto será CNN 1D (comentar el que no se utilice). Considerar que se debe usar el mismo modelo que usa el cliente.
```
    #modelo cnn1D
    model = create_CNN1D_model(n_timesteps, n_features, n_outputs)

    #modelo DNN
    #model = create_DNN_model(n_timesteps,n_features,n_outputs)
```
- Al definir una estrategia se debe indicar el número mínimo de clientes usados durante el entrenamiento (por ronda), número mínimo de clientes en el sistema para que se comience el entrenamiento, por defecto estos parámetros son 10 los dos.
    <img src="https://i.imgur.com/EnGoCLl.png">
- Por último se debe indicar el puerto (```0.0.0.0:7000```, por defecto), el cual tiene que ser el mismo que el de los clientes, y el número de rondas que se requieran, por defecto estas son 20.
    <img src="https://i.imgur.com/to65ICs.png">
- Ahora que el server está listo, ya se puede ejecutar: ```python server.py```, esperar unos segundos y ya se puede ejecutar algunos clientes.
### Ejecutar cliente <a name="id2"></a>
- Definir el modelo, se debe cambiar según se requiera, por defecto será CNN 1D (comentar el que no se utilice). Considerar que se debe usar el mismo modelo que usa el servidor.
```
    #modelo cnn1D
    model = create_CNN1D_model(n_timesteps, n_features, n_outputs)

    #modelo DNN
    #model = create_DNN_model(n_timesteps,n_features,n_outputs)
```
- En la función fit se pueden definir el número de épocas que se requiere que haga el cliente en el entrenamiento local, por defecto este parametro es 5.
    <img src="https://i.imgur.com/fdaCdEL.png">
- Por último se debe indicar el puerto (```0.0.0.0:7000```, por defecto), el cual tiene que ser el mismo que el del servidor.
    <img src="https://i.imgur.com/jHcfJ9m.png">
 - Para ejecutar el cliente, se debe tener en cuenta los datos que se le quieran pasar desde el conjunto de datos, por ejemplo el cliente 1 tendrá los datos de 2 sujetos de entrenamiento del conjunto de datos, del sujeto 1 y del sujeto 3, Entonces se ejectua de la siguiente manera: ```python client.py 1 3```.
En cambio el cliente 10, tiene los datos de 3 sujetos del conjunto de datos, el sujeto 28, 29 y 30, por lo que se ejecuta de la siguiente manera ```python client.py 28 30```. El conjunto de datos que se le asignen a cada cliente, es un rango entre el primer dato del primer sujeto y el último dato del segundo sujeto. Sólo pueden ser utilizados los sujetos del grupo de entrenamiento del dataset, los cuales son 1,  3,  5,  6,  7,  8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29 y 30.

- Para mayor simplicidad ejecutar el archivo ```run_10_clients.sh```, con ```bash run_10_clients.sh```, el cual ejecuta 10 clientes (donde cada cliente tiene los datos de 2 sujetos del conjunto de datos de entrenamiento).
### Salida <a name="id3"></a>
- Se puede apreciar en el recuadro rojo la precisión del modelo global en cada ronda, y en el recuadro azul se ve la precisión que alcanzó cada cliente con su modelo local en la última ronda. Para este ejemplo participaron 10 clientes con 20 rondas de comunicación.
    <img src="https://i.imgur.com/oRqXnqn.png">    

### Referencias
- Dataset: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
- Preproceso: https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
