from sklearn.svm import SVC

NOMBRE = "SVM"

def crear_modelo() -> SVC:
    return SVC(class_weight='balanced')