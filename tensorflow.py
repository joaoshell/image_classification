

#Instalar biblioteca

pip install tensorflow

#Importando bibliotecas

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 #escalando os valores no intervalo de 0 a 1 através da divisão por 255
                                                  # valor dos possiveis para cada pixel

#Construindo o modelo

#Construir a rede neural requer configurar as camadas do modelo, e depois, compilar o modelo. Muito do deep #learning consiste em encadear simples camadas


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), #transforma o formato da imagem de um array 2x2 para 1x
    tf.keras.layers.Dense(128, activation='relu'), #camarada neural. Pode ser densely connected ou fully connected. 128 significa o número de neuronios
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax') #camada neural. Softmax de 10 nós retornando um array de 10 probabilidades, onda a soma resulta em 1
])

#Compilando o modelo

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Função Loss — Mede quão precisa o modelo é durante o treinamento. Queremos minimizar a função para guiar o modelo para a direção certa.
# Optimizer —Isso é como o modelo se atualiza com base no dado que ele vê e sua função loss.
# Métricas —usadas para monitorar os passos de treinamento e teste. O exemplo abaixo usa a acurácia, a fração das imagens que foram classificadas corretamente.

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test, verbose=2) #verbose 0 = silent, 1 = progress bar = single line

#Overfitting é quando um modelo de aprendizado de máquina performou de maneira pior em um conjunto de entradas novas, e não usadas anteriormente, que usando o conjunto de treinamento.

