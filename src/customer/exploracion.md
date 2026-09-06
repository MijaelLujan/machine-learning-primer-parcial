# Inicio

Primero vamos a ver que tiene el dataset:

```python
df = pl.read_csv("../data/Customer.csv")
df
```

# Descripcion del dataset
Vemos que tiene un shape de 793 rows X 9 columns ( 9 features ), con el primer vistaso, vemos atributo por atributo:
- CustomerID: Codigos, tiene dos letras seguido de numeros. Estas dos primeras letras pareciera ser las iniales de CustomerName. Pareciera ser una variable dependiente
- CustomerName: Nombre de los clientes pareciera ser que solo es un nombre y un apellido
- Segment: Pareciera ser una clasificacion del tipo de cliente
- Age: Edad del cliente
- Country: Pais de los clientes
- City: ciudad de los clientes
- State: Estado (de eeuu pareciera) de los clientes
- PostalCode: codigo postal del cliente
- Region: clasificacion de donde queda.

Ahora vamos a ver si existe campos nullos, blancos o malformados.

```python
conteo_nulos = df.null_count()
conteo_nulos
```


no tenemos ningun nulo en el dataset. entonces ahora veremos, que columnas son verdaderamente categoricas, estas son:
- Segment
- Country
- City
- State
- PostalCode
- Region


```python
def sacar_lista_unicos_columnas(df, list_column_names) -> dict[str , list[str]]:
    resp = {}
    for column in list_column_names:
        resp[column] = df[column].unique().to_list()
    return resp
sacar_lista_unicos_columnas(df, ["Segment", "Country", "City", "State", "Postal Code", "Region"])
```

Vemos, varias cosas, no existe valores blancos, detallamos informacion encontrada:
- Segment: solo tiene 3 'Corporate', 'Home Office', 'Consumer'
- Country: Es solo de EEUU
- Region: Solo tiene 4, ['South', 'Central', 'East', 'West'].
Por el momento las otras columnas tenemos que obtener mas datos, una aproximacion para ver datos malformado seria contar los estados si superan mas de 50, ciudades que tambien superan, como los codigos postales que tiene eeuu son 41500, supera por mucho la cantidad de rows, Pero vamos a ver cuantos existen.

```python
def sacar_conteo_unicos_columnas(df, list_column_names) -> dict[str , int]:
    resp = {}
    for column in list_column_names:
        resp[column] = len(df[column].unique().to_list())
    return resp
sacar_conteo_unicos_columnas(df, ["City", "State", "Postal Code"])
```

Tenemos 41 estados, 252 ciudades y 314 codigos postales, entonces son todos validos estos. 

Verificaremos datos repetidos con el CustomerId.

```python
sacar_conteo_unicos_columnas(df, ["Customer ID", "Customer Name"])
```

Tenemos como 793, no existe datos repetidos ni nombres repetidos.


Verificamos si los codigos de Customer Name solo existe dos palabras, luego vamos a verificar la hipotesis de que las dos letras de Customer ID son las iniciales de estas.

```python

def cantidad_palabras(nombre: str):
    nombre = nombre.strip()
    conteo = len(nombre.split())
    return conteo

custumer_conteo = df["Customer Name"].map_elements(cantidad_palabras).unique().to_list()
custumer_conteo

```

Los resultados muestran que existe 3 palabras en las columnas entonces puede que exista un solo nombre, o hasta 3 Palabras por nombre. Igualmente vamos a verificar las letras de ID.


```python
def cantidad_letras(nombre: str):
    nombre = nombre.strip()
    conteo = len(nombre.split("-")[0])
    return conteo

custumer_id_conteo = df["Customer ID"].map_elements(cantidad_letras).unique().to_list()
custumer_id_conteo
```

Siempre son dos letras en el Customer ID, vamos a buscar las rows donde aparezcan 1, y 3 palabras para ver relacion.


```python

customers_filtrados = df.filter(
    df["Customer Name"].map_elements(cantidad_palabras).is_in([1, 3])
)
customers_filtrados
```

Son solo 6 datos, quizas no son tan importantes, aun asi. En caso de 3 palabras se toma las primeras 2 iniciales para Customer ID, y en el cas de una palabra, es el unico caso que pone el codigo como Mayuscula la primera letra y 2da letra como minuscula el dato es: "Co-12640"	"Corey-Lock"

Verificamos como ultimo la edad, si existe valores negativos.


```python

conteo_negativos = df.filter(pl.col("Age") < 0).height
conteo_negativos
```

No existe edades negativas por lo tanto no se encontro datos malformados en el dataset. como polars intuye los tipos de las columnas automaticamente, no se necesita verficar si esta mezclado ya que hubiera aparecido como dato OBJECT, en su lugar si reconocio String, i64.