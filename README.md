Estudo de Redes Neurais com PyTorch 🧠🚀


Este repositório contém um projeto de estudo prático focado nos fundamentos de redes neurais. O objetivo é construir e treinar um modelo simples, do zero ✨, usando a biblioteca PyTorch para classificar os dígitos do famoso dataset MNIST.

Conceitos Abordados 💡
Classes e Objetos 🏗️

Classe (nn.Module): No PyTorch, nn.Module é a classe base, agindo como um "molde" ou projeto para a criação de redes neurais. Ela fornece a estrutura e as ferramentas necessárias. 💻

Objeto (Modelo): É uma instância real de uma rede neural, criada a partir do molde nn.Module. Seu código define o objeto Modelo com camadas específicas para o problema de classificação. 🤖

Camadas e Funções de Ativação 🔌

As camadas (nn.Linear) são as "propriedades" do seu modelo. Elas definem a estrutura da rede, como o número de neurônios e as conexões entre eles. 🧱

ReLU e log_softmax são as funções de ativação. Elas agem como "métodos" que processam os dados nas camadas, introduzindo a não-linearidade necessária para que a rede possa aprender padrões complexos. ✨

Fluxo de Dados (forward) ➡️

O método forward define o fluxo de dados através do modelo. Ele é o coração do encapsulamento, pois descreve como as camadas e funções de ativação interagem para transformar os dados de entrada em previsões de saída. 🧠

Treinamento e Validação 📈

O código inclui a lógica para o treinamento, onde o modelo aprende com os dados 🤓, e a validação, onde o desempenho do modelo é avaliado com dados que ele nunca viu. Isso garante que a rede não memorize apenas o conjunto de treino. ✅

Estudo feito pelo DIO.me
