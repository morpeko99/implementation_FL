#!/bin/bash

sleep 2

#solo se utilizan los datos de los sujetos de entrenamiento 1,  3,  5,  6,  7,  8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30

#cada cliente tendrá los datos de 4 sujetos de entrenamiento (exceptuando el último cliente que tendrá datos de 5 sujetos de entrenamiento)

python client.py 1 3 & #contiene los datos de los sujetos 1,  3,  
python client.py 5 6 & #contiene los datos de los sujetos 5,  6
python client.py 7 8 & #contiene los datos de los sujetos 7, 8
python client.py 11 14  & #contiene los datos de los sujetos 11, 14
python client.py 15 16 & #contiene los datos de los sujetos 15, 16
python client.py 17 19 & #contiene los datos de los sujetos 15, 16
python client.py 21 22 & #contiene los datos de los sujetos 17 19 
python client.py 23 25 & #contiene los datos de los sujetos 23 25
python client.py 26 27 & #contiene los datos de los sujetos 26 27
python client.py 28 30 & #contiene los datos de los sujetos 28 29 30

# Use CTRL+C para parar todos los procesos
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
sleep 86400
