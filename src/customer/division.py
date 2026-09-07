from pathlib import Path
import polars as pl

# Definición de rutas relativas
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PATH_RAW = DATA_DIR / "Customer.csv"
PATH_CLEAN = DATA_DIR / "Customer_clean.csv"

from sklearn.model_selection import train_test_split
def dividir_datos(df: pl.DataFrame):
    df_pandas = df.to_pandas()

    train, test = train_test_split(df_pandas,test_size=0.30,random_state=42)
    train = pl.from_pandas(train)
    test = pl.from_pandas(test)

    return train, test


train, test = dividir_datos(df_limpio)
print("Entrenamiento:", train.height)
print("Pruebas:", test.height)