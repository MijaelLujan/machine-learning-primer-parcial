from sklearn.linear_model import LogisticRegression

NOMBRE = "Regresion Logistica"

def crear_modelo() -> LogisticRegression:
    return LogisticRegression(max_iter=1000)