"""
Modelo de red neuronal mas simple posible para la clasificacion de basura.
Este modelo constara de una sola capa oculata donde el numero de neuronas sera igual
al numero de claseas, que tiene el conjunto de datos un total de 6 clases.
Luego se realizara un GlobalAvaregePooling o un GlobalMazPooling para asi resucir a un vector
de 6 elemento es decir igual a la clases que tenemos.
"""