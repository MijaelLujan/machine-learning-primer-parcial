from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, precision_recall_fscore_support
)
from sklearn.utils.class_weight import compute_sample_weight

from customer.modelos import MODULOS


def comparar_modelos(input_path: Path, target_col: str) -> pd.DataFrame:
    print(f"Cargando datos desde: {input_path} (target={target_col})")
    df = pd.read_csv(input_path)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)

    filas = []
    for modulo in MODULOS:
        modelo = modulo.crear_modelo()

        if modulo.NOMBRE == "Perceptron Multicapa":
            modelo.fit(X_train, y_train)
        else:
            modelo.fit(X_train, y_train, sample_weight=sample_weight)

        y_pred = modelo.predict(X_test)
        labels = sorted(y_test.unique())

        graficar_matriz_confusion(y_test, y_pred, modulo.NOMBRE, labels, target_col)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        filas.append({
            'Modelo': modulo.NOMBRE,
            'Accuracy': acc,
            'Balanced': bal_acc,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
        })

        print(f"\n{modulo.NOMBRE} (accuracy={acc:.4f}):")
        print(classification_report(y_test, y_pred, zero_division=0))

    tabla = pd.DataFrame(filas).sort_values('Accuracy', ascending=False)
    print(f"\nComparacion final ({target_col}):")
    print(tabla.to_string(index=False))

    return tabla


def graficar_matriz_confusion(y_test, y_pred, nombre_modelo, labels, target_col):
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Matriz de Confusion - {nombre_modelo} ({target_col})')
    plt.xlabel('Prediccion')
    plt.ylabel('Real')
    plt.tight_layout()

    output_dir = Path(__file__).resolve().parent / 'Matriz de confusion'
    output_dir.mkdir(exist_ok=True)
    nombre_archivo = f'matriz_confusion_{target_col}_{nombre_modelo.replace(" ", "_")}.png'
    plt.savefig(output_dir / nombre_archivo)
    plt.close()


def _resolver_project_root() -> Path:
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / 'pyproject.toml').exists():
            return parent
    return current_file.parent


if __name__ == '__main__':
    project_root = _resolver_project_root()
    data_dir = project_root / 'data'

    tabla_segment = comparar_modelos(
        data_dir / 'Customer_segment_preprocessed.csv',
        target_col='target_segment'
    )
    #tabla_region = comparar_modelos(
    #    data_dir / 'Customer_region_preprocessed.csv',
    #    target_col='target_region'
    #)