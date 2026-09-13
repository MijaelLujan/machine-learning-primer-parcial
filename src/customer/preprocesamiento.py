"""
Módulo de Preprocesamiento de Datos - Customer Dataset
Aplica estandarización a columnas numéricas y vectorización (One-Hot) a categóricas.
"""

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def preprocess_customer_data(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Lee Customer_clean.csv, aplica transformaciones y guarda el resultado preprocesado."""
    print(f"Cargando datos desde: {input_path}")
    df = pd.read_csv(input_path)

    # 1. Definición de variables
    num_cols = ['age']
    cat_cols = ['segment', 'region']

    # 2. Configuración del preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ],
        remainder='drop',
    )

    # 3. Aplicar transformación
    X_processed = preprocessor.fit_transform(df)

    # 4. Estructurar DataFrame con nombres de columnas
    feature_names = preprocessor.get_feature_names_out()
    df_processed = pd.DataFrame(X_processed, columns=feature_names)

    # 5. Exportar CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    print(f"Archivo guardado exitosamente en: {output_path}")
    print(f"Dimensiones finales: {df_processed.shape}")
    print("\nPrimeras 5 filas:")
    print(df_processed.head())

    return df_processed


if __name__ == '__main__':
    # Localiza la raíz del proyecto sin importar desde dónde se invoque el script
    current_file = Path(__file__).resolve()
    # Si guardas este archivo dentro de 'src/', sube un nivel a la raíz del proyecto:
    project_root = current_file.parent.parent if current_file.parent.name == 'src' else current_file.parent

    input_file = project_root / 'data' / 'Customer_clean.csv'
    output_file = project_root / 'data' / 'Customer_preprocessed.csv'

    preprocess_customer_data(input_file, output_file)