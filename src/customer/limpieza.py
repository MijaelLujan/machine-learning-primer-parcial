"""
Módulo de limpieza y preparación de datos para Customer.
Resumen de Transformaciones:
- Volumen: 793 filas y 8 columnas finales (sin pérdida de registros).
- Nomenclatura: Conversión a snake_case sin espacios (ej. customer_id, postal_code).
- Limpieza de Texto: Eliminación de espacios residuales (strip_chars) en columnas categóricas.
- Consistencia de IDs: Conversión a mayúsculas para corregir inconsistencias (ej. 'Co-12640' -> 'CO-12640').
- Tipos de Datos: 'postal_code' casteado a String (código territorial no numérico).
- Selección de Features: Descarte de 'country' por varianza cero (100% United States).
- Duplicados/Nulos: Validación de 0 registros nulos y 0 duplicados en 'customer_id'.
- Salida: data/Customer_clean.csv
"""

from pathlib import Path
import polars as pl

# Definición de rutas relativas
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PATH_RAW = DATA_DIR / "Customer.csv"
PATH_CLEAN = DATA_DIR / "Customer_clean.csv"


def cargar_datos(ruta: Path = PATH_RAW) -> pl.DataFrame:
    """Carga el dataset Customer en formato CSV."""
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo en: {ruta}")
    return pl.read_csv(ruta)


def estandarizar_columnas(df: pl.DataFrame) -> pl.DataFrame:
    """Normaliza los nombres de columnas a snake_case sin espacios."""
    mapeo = {col: col.strip().lower().replace(" ", "_") for col in df.columns}
    return df.rename(mapeo)


def limpiar_datos(df: pl.DataFrame) -> pl.DataFrame:
    """
    Ejecuta el pipeline de limpieza:
    - Normalización de encabezados.
    - Limpieza de espacios en strings (strip_chars).
    - Unificación a mayúsculas de IDs (corrige 'Co-12640' -> 'CO-12640').
    - Casteo de postal_code a String.
    - Descarte de duplicados por customer_id.
    - Eliminación de columna country (varianza cero).
    """
    # 1. Normalizar columnas
    df_clean = estandarizar_columnas(df)

    # 2. Corrección de tipos y limpieza de texto
    df_clean = df_clean.with_columns(
        [
            pl.col("customer_id").str.to_uppercase().str.strip_chars(),
            pl.col("customer_name").str.strip_chars(),
            pl.col("segment").str.strip_chars(),
            pl.col("city").str.strip_chars(),
            pl.col("state").str.strip_chars(),
            pl.col("region").str.strip_chars(),
            pl.col("postal_code").cast(pl.String),
        ]
    )

    # 3. Eliminar duplicados si existieran
    df_clean = df_clean.unique(subset=["customer_id"])

    # 4. Descartar columna 'country' (solo contiene 'United States')
    if "country" in df_clean.columns:
        df_clean = df_clean.drop("country")

    return df_clean


def guardar_datos(df: pl.DataFrame, ruta_salida: Path = PATH_CLEAN) -> None:
    """Exporta el DataFrame procesado a un archivo CSV."""
    df.write_csv(ruta_salida)
    print(f"[OK] Archivo limpio generado en: {ruta_salida}")


def main():
    print("Iniciando pipeline de limpieza...")
    df_raw = cargar_datos()
    print(f"Filas originales: {df_raw.height}")

    df_limpio = limpiar_datos(df_raw)
    print(f"Filas limpias: {df_limpio.height}")
    print(f"Columnas finales ({len(df_limpio.columns)}): {df_limpio.columns}")

    guardar_datos(df_limpio)


if __name__ == "__main__":
    main()