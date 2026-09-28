from sklearn.neural_network import MLPClassifier

NOMBRE = "Perceptron Multicapa"

def crear_modelo() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=2000,
        random_state=42,
    )