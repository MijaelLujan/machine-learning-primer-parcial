from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def preprocess_region_classification(
    input_path: Path, output_path: Path
) -> pd.DataFrame:
  """Preprocesa las variables geograficas (X: country, state, city, zip)

  para predecir la macrozona (y: region), valida splits y exporta el CSV final.
  """
  print(f"Cargando datos desde: {input_path}")
  df = pd.read_csv(input_path)

  if 'postal_code' in df.columns and 'zip' not in df.columns:
    df['zip'] = df['postal_code'].astype(str)
  elif 'zip' in df.columns:
    df['zip'] = df['zip'].astype(str)


  expected_features = ['country', 'state', 'city', 'zip']
  available_features = [col for col in expected_features if col in df.columns]

  X = df[available_features]
  y = df['region']

  # 3. Configuracion del transformador categorico
  preprocessor = ColumnTransformer(
      transformers=[(
          'cat',
          OneHotEncoder(handle_unknown='ignore', sparse_output=False),
          available_features,
      )]
  )

  # 4. Validacion de particion train/test estratificada (80/20)
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  X_train_processed = preprocessor.fit_transform(X_train)
  X_test_processed = preprocessor.transform(X_test)

  print(f"Dimensiones de X_train procesado: {X_train_processed.shape}")
  print(f"Dimensiones de X_test procesado: {X_test_processed.shape}")
  print(f"\nDistribucion de clases en target (region):\n{y.value_counts()}")


  X_all_processed = preprocessor.fit_transform(X)
  feature_names = preprocessor.get_feature_names_out()

  df_processed = pd.DataFrame(X_all_processed, columns=feature_names)
  df_processed['target_region'] = y.values


  output_path.parent.mkdir(parents=True, exist_ok=True)
  df_processed.to_csv(output_path, index=False)

  print(f"\nArchivo guardado exitosamente en: {output_path}")
  print(f"Dimensiones finales: {df_processed.shape}")
  print("\nPrimeras 3 filas:")
  print(df_processed.head(3))

  return df_processed


if __name__ == '__main__':
  current_file = Path(__file__).resolve()
  project_root = (
      current_file.parent.parent
      if current_file.parent.name in ['src', 'customer']
      else current_file.parent
  )

  input_file = project_root / 'data' / 'Customer_clean.csv'
  output_file = project_root / 'data' / 'Customer_region_preprocessed.csv'

  preprocess_region_classification(input_file, output_file)