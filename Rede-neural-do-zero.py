import numpy as np               # Importa a biblioteca NumPy para operações numéricas.
import torch                     # Importa a biblioteca principal do PyTorch.
import torch.nn.functional as F  # Importa o módulo funcional do PyTorch (para ReLU, softmax, etc.).
import torchvision               # Importa a biblioteca torchvision, usada para datasets e transformações.
import matplotlib.pyplot as plt  # Importa a biblioteca Matplotlib para visualização de gráficos.
from time import time            # Importa a função time para medir o tempo de execução.
from torchvision import datasets, transforms # Importa módulos específicos de datasets e transformações.
from torch import nn, optim      # Importa o módulo de redes neurais e o otimizador.


transform = transforms.ToTensor() # Cria um objeto de transformação para converter imagens para tensores.


trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform) # Carrega o dataset de treino MNIST e aplica a transformação.
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=64)       # Cria um "carregador de dados" para iterar sobre o conjunto de treino.


valset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform) # Carrega o dataset de validação.
valloader = torch.utils.data.DataLoader(valset, shuffle=True, batch_size=64)           # Cria um "carregador de dados" para o conjunto de validação.

print("Conjuntos de dados MNIST carregados com sucesso!") # Imprime uma mensagem de sucesso.


transform = transforms.ToTensor() # A linha se repete, mas continua com a mesma função.


trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform) # Repetição.
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=64)       # Repetição.


valset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform) # Repetição.
valloader = torch.utils.data.DataLoader(valset, shuffle=True, batch_size=64)           # Repetição.

print("Conjuntos de dados MNIST carregados com sucesso!") # Repetição.

# O código abaixo precisa ser executado após uma iteração do 'trainloader' para que 'imagens' e 'etiquetas' existam.
# A seguir, o código verifica o tamanho dos tensores de imagem e etiqueta.
print(imagens[0].shape)  # Imprime as dimensões do tensor da primeira imagem do lote.
print(etiquetas[0].shape) # Imprime as dimensões do tensor da primeira etiqueta do lote.

# O código se repete, mas a função é a mesma.
print(imagens[0].shape)  # Imprime as dimensões do tensor da primeira imagem do lote.
print(etiquetas[0].shape) # Imprime as dimensões do tensor da primeira etiqueta do lote.

# A seguir está a estrutura de treinamento do modelo.
#
# A seguir, o código é a estrutura de treinamento do modelo.
#
# O bloco abaixo é onde ocorre o treinamento do modelo.
#-----------------------------------------------------------------------------------------------------------------
import torch.nn as nn                                          # Importa o módulo de redes neurais.
import torch.optim as optim                                      # Importa o módulo de otimizadores.
from time import time                                          # Importa a função de tempo.

# A função de treinamento agora tem a estrutura correta.
def treino(modelo, trainloader, device):                       # Define a função de treino com o modelo, carregador e dispositivo.
    otimizador = optim.SGD(modelo.parameters(), lr=0.01, momentum=0.5) # Define o otimizador SGD para atualizar os pesos do modelo.
    inicio = time()                                              # Inicia a contagem do tempo de treino.
    criterio = nn.NLLLoss()                                      # Define a função de perda (critério) para medir o erro do modelo.
    EPOCHS = 10                                                  # Define o número de épocas (passagens completas pelo dataset).
    modelo.train()                                               # Coloca o modelo em modo de treino (ativa funcionalidades como o dropout).

    # Laço principal para as épocas de treinamento.
    for epoch in range(EPOCHS):                                  # Inicia o loop para cada época.
        perda_acumulada = 0.0                                    # Zera a perda acumulada para cada nova época.
        
        # Laço interno para iterar sobre os dados do DataLoader.
        for imagens, etiquetas in trainloader:                   # Itera sobre os lotes de imagens e etiquetas do DataLoader.
            # Reorganiza o formato das imagens para um vetor de uma dimensão (784 pixels).
            imagens = imagens.view(imagens.shape[0], -1)         # Transforma o tensor da imagem 3D em 2D, achatando os pixels.

            # Zera os gradientes do otimizador antes de cada passo de backpropagation.
            otimizador.zero_grad()                               # Limpa os gradientes acumulados para o novo cálculo.

            # Passagem para frente (forward pass).
            output = modelo(imagens.to(device))                  # Envia as imagens para o modelo para obter as previsões.
            
            # Calcula a perda da rede.
            perda_instantanea = criterio(output, etiquetas.to(device)) # Compara as previsões com as etiquetas corretas e calcula a perda.

            # Passagem para trás (backpropagation).
            perda_instantanea.backward()                         # Calcula os gradientes da perda em relação aos pesos do modelo.
            
            # Atualiza os pesos da rede.
            otimizador.step()                                    # Usa os gradientes para ajustar os pesos e otimizar a rede.

            # Acumula a perda de cada lote.
            perda_acumulada += perda_instantanea.item()          # Soma a perda do lote à perda total da época.

        # Imprime o resultado de perda para a época atual.
        print("Epoch {} - Perda resultante: {}".format(epoch + 1, perda_acumulada / len(trainloader))) # Calcula a perda média e a imprime.

    # Imprime o tempo total de treino no final.
    print("Tempo de treino (em minutos) = ", (time() - inicio) / 60) # Calcula e imprime o tempo total de treinamento.
#-----------------------------------------------------------------------------------------------------------------
# Abaixo, a estrutura de validação do treinamento.
def validacao(modelo, valloader, device):                      # Define a função para validar o modelo.
    conta_corretas, conta_todas = 0, 0                           # Inicializa os contadores de acertos e de total de imagens.
    for imagens, etiquetas in valloader:                         # Itera sobre os lotes do DataLoader de validação.
        for i in range(len(etiquetas)):                          # Itera sobre cada imagem no lote.
            img = imagens[i].view(1, 784)                        # Acha a imagem para passar uma de cada vez.
            # desabilita a propagacao dos gradientes.
            with torch.no_grad():                                # Desativa o cálculo de gradientes para otimizar a memória.
                logps = modelo(img.to(device))                   # Passagem para frente, obtendo os logaritmos de probabilidade.

            ps = torch.exp(logps)                                # Converte os logaritmos de probabilidade em probabilidades.
            probab = list(ps.cpu().numpy()[0])                   # Converte o tensor de probabilidades em uma lista.
            etiqueta_pred = probab.index(max(probab))            # Encontra o índice (classe) com a maior probabilidade.
            etiqueta_certa = etiquetas.numpy()[i]                # Obtém a etiqueta correta da imagem.
            if(etiqueta_certa == etiqueta_pred):                 # Compara a previsão com a etiqueta real.
                conta_corretas += 1                              # Se estiverem corretas, incrementa o contador de acertos.
            conta_todas += 1                                     # Incrementa o contador total de imagens testadas.
        
    print("Total de imagens testadas: ", conta_todas)            # Imprime o total de imagens testadas.
    print("\nPrecisção do modelo = {}%", format(conta_corretas*100/conta_todas)) # Imprime a precisão final do modelo.

# Abaixo, o código para instanciar o modelo e definir o dispositivo.
modelo = Modelo()                                              # Cria uma instância da classe Modelo.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Verifica se há uma GPU disponível, senão usa a CPU.
modelo.to(device)                                                # Move o modelo para o dispositivo selecionado (GPU ou CPU).