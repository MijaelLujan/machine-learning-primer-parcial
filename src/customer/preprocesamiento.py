from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def preprocess_customer_data(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Lee Customer_clean.csv, extrae la zona postal, aplica transformaciones y exporta."""
    print(f"Cargando datos desde: {input_path}")
    df = pd.read_csv(input_path)

    # 1. Feature Engineering: Extraer macro-zona del código postal
    df['postal_zone'] = df['postal_code'].astype(str).str[0]

    # 2. Definición de variables
    num_cols = ['age']
    cat_cols = ['segment', 'region', 'postal_zone']

    # 3. Configuración del preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ],
        remainder='drop',  # Descarta customer_id, customer_name, postal_code original, city, etc.
    )

    # 4. Ajustar y transformar
    X_processed = preprocessor.fit_transform(df)

    # 5. Reconstruir DataFrame con nombres de columnas
    feature_names = preprocessor.get_feature_names_out()
    df_processed = pd.DataFrame(X_processed, columns=feature_names)

    # 6. Exportar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    
    print(f"Archivo guardado exitosamente en: {output_path}")
    print(f"Dimensiones finales: {df_processed.shape}")
    print("\nPrimeras 5 filas:")
    print(df_processed.head())

    return df_processed


if __name__ == '__main__':
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent if current_file.parent.name == 'src' else current_file.parent

    input_file = project_root / 'data' / 'Customer_clean.csv'
    output_file = project_root / 'data' / 'Customer_preprocessed.csv'

    preprocess_customer_data(input_file, output_file)