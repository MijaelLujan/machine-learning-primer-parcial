import polars as pl

df = pl.read_csv("../data/Customer.csv")
conteo_nulos = df.null_count()


def sacar_lista_unicos_columnas(df, list_column_names) -> dict[str, list[str]]:
    resp = {}
    for column in list_column_names:
        resp[column] = df[column].unique().to_list()
    return resp


sacar_lista_unicos_columnas(
    df, ["Segment", "Country", "City", "State", "Postal Code", "Region"]
)


def sacar_conteo_unicos_columnas(df, list_column_names) -> dict[str, int]:
    resp = {}
    for column in list_column_names:
        resp[column] = len(df[column].unique().to_list())
    return resp


sacar_conteo_unicos_columnas(df, ["City", "State", "Postal Code"])
sacar_conteo_unicos_columnas(df, ["Customer ID", "Customer Name"])


def cantidad_palabras(nombre: str):
    nombre = nombre.strip()
    conteo = len(nombre.split())
    return conteo


custumer_conteo = df["Customer Name"].map_elements(cantidad_palabras).unique().to_list()


def cantidad_letras(nombre: str):
    nombre = nombre.strip()
    conteo = len(nombre.split("-")[0])
    return conteo


custumer_id_conteo = df["Customer ID"].map_elements(cantidad_letras).unique().to_list()
customers_filtrados = df.filter(
    df["Customer Name"].map_elements(cantidad_palabras).is_in([1, 3])
)

conteo_negativos = df.filter(pl.col("Age") < 0).height
