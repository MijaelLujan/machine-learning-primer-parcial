from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from customer.modelos import MODULOS

def comparar_modelos(input_path: Path) -> dict:
  print(f"Cargando datos desde: {input_path}")
  df = pd.read_csv(input_path)

  X = df.drop(columns=['target_segment'])
  y = df['target_segment']

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  resultados = {}
  for modulo in MODULOS:
    modelo = modulo.crear_modelo()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    resultados[modulo.NOMBRE] = acc

    print(f"\n=== {modulo.NOMBRE} (accuracy={acc:.4f}) ===")
    print(classification_report(y_test, y_pred))

  print("\n=== Comparacion final ===")
  for nombre, acc in sorted(resultados.items(), key=lambda x: -x[1]):
    print(f"{nombre}: {acc:.4f}")

  return resultados


def _resolver_project_root() -> Path:
  current_file = Path(__file__).resolve()
  for parent in current_file.parents:
    if (parent / 'pyproject.toml').exists():
      return parent
  return current_file.parent


if __name__ == '__main__':
  project_root = _resolver_project_root()
  input_file = project_root / 'data' / 'Customer_segment_preprocessed.csv'
  comparar_modelos(input_file)