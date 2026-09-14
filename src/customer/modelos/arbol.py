from sklearn.tree import DecisionTreeClassifier

NOMBRE = "Arbol de Decision"

def crear_modelo() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=42)