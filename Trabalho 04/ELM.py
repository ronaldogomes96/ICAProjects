import numpy as np


class ELMClassifier:

    def __init__(self, hiddenSize):
        self.bias = None
        self.hiden_weights = None
        self.output_weights = None
        self.hiddenSize = hiddenSize


    def fit(self, X, y):
        # Inicializar matriz de pesos W
        a, b = -0.5, 0.5
        self.hiden_weights = a + (b - a) * np.random.rand(self.hiddenSize, X.shape[1] + 1)

        # Calcular saídas da camada oculta para o conjunto de treinamento
        Z = []
        for t in range(X.shape[0]):
            X_ocult = np.concatenate(([1], X[t]))
            Ui = np.dot(self.hiden_weights, X_ocult)
            Yi = (1 - np.exp(-Ui)) / (1 + np.exp(-Ui))
            Z.append(Yi)
        Z = np.array(Z)

        self.output_weights = np.dot(np.linalg.pinv(Z), y.to_numpy().reshape(-1, 1)).T

    def predict(self, X):
        # Calcular saídas da camada oculta para o conjunto de teste
        Z1 = []
        for t in range(X.shape[0]):
            X_hidden = np.concatenate(([1], X[t]))
            Ui = np.dot(self.hiddenSize, X_hidden)
            Yi = (1 - np.exp(-Ui)) / (1 + np.exp(-Ui))
            Z1.append(Yi)
        Z1 = np.array(Z1)

        # Gerar saída da rede para todos os vetores de entrada de teste
        Y = np.dot(self.output_weights, Z1)
        Y_teste = np.sign(Y)
        return np.argmax(Y_teste)
