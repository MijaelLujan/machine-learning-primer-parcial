# Informe Técnico: Preprocesamiento de Datos para Machine Learning
**Dataset:** `Customer_clean.csv`  
**Notebook:** `02_limpieza_customerNicole.ipynb`  
**Fase:** Ingeniería de Características y Preparación de Datos (Feature Engineering & Preprocessing)

---

## 1. Resumen Ejecutivo
Se realizó la etapa de preprocesamiento sobre el conjunto de datos de clientes (`Customer_clean.csv`, 793 registros y 8 columnas), transformando los atributos brutos en una matriz numérica optimizada y estandarizada, apta para ser consumida directamente por algoritmos de Machine Learning (modelos de regresión, clasificación o algoritmos de clustering tipo K-Means).

---

## 2. Diagnóstico Inicial del Dataset (`df.info()`)
El dataset original cuenta con 793 filas sin valores nulos (100% de completitud) distribuidas en las siguientes columnas:
* `customer_id` (str): Identificador alfanumérico único del cliente.
* `customer_name` (str): Nombre del cliente.
* `segment` (str): Segmento de mercado (Consumer, Corporate, Home Office).
* `age` (int64): Edad del cliente (rango continuo).
* `city` (str): Ciudad de residencia.
* `state` (str): Estado geográfico.
* `postal_code` (int64): Código postal (variable categórica codificada numéricamente).
* `region` (str): Región geográfica macro (Central, East, South, West).

---

## 3. Criterios de Selección y Descarte de Variables

1. **Variables Descartadas (`remainder='drop'`):**
   * `customer_id` y `customer_name`: Identificadores únicos sin capacidad de generalización predictiva (evita sobreajuste y ruido).
   * `postal_code` y `city`: Variables geográficas de alta cardinalidad que generarían una dispersión dimensional excesiva para el volumen de datos disponible (793 filas).

2. **Variable Numérica Seleccionada:**
   * `age`: Atributo demográfico continuo de relevancia analítica.

3. **Variables Categóricas Seleccionadas:**
   * `segment` (3 niveles): Atributo conductual/comercial.
   * `region` (4 niveles): Atributo espacial generalizado.

---

## 4. Transformaciones Aplicadas (`ColumnTransformer`)

Para garantizar reproducibilidad y evitar fuga de datos (*data leakage*), se implementó un `ColumnTransformer` de `scikit-learn` compuesto por:

### A. Estandarización Numérica (`StandardScaler`)
* **Columna tratada:** `age`
* **Método:** Transformación a puntajes Z ($Z = \frac{x - \mu}{\sigma}$), centrando la media en 0 y fijando la desviación estándar en 1.
* **Resultado:** La columna pasa a llamarse `num__age`. Se eliminan las diferencias de escala respecto a variables binarias, asegurando convergencia óptima en modelos basados en distancias o gradientes.

### B. Vectorización Categórica (`OneHotEncoder`)
* **Columnas tratadas:** `segment`, `region`
* **Configuración:** `handle_unknown='ignore'`, `sparse_output=False`
* **Resultado:** Codificación one-hot que crea variables binarias (0/1):
  * `segment` → `cat__segment_Consumer`, `cat__segment_Corporate`, `cat__segment_Home Office`
  * `region` → `cat__region_Central`, `cat__region_East`, `cat__region_South`, `cat__region_West`

---

## 5. Estructura de la Matriz Resultante
* **Dimensiones finales:** 793 filas × 8 características numéricas.
* **Composición de columnas generadas:**
  1. `num__age` (Continuo normalizado)
  2. `cat__segment_Consumer` (Binario 0/1)
  3. `cat__segment_Corporate` (Binario 0/1)
  4. `cat__segment_Home Office` (Binario 0/1)
  5. `cat__region_Central` (Binario 0/1)
  6. `cat__region_East` (Binario 0/1)
  7. `cat__region_South` (Binario 0/1)
  8. `cat__region_West` (Binario 0/1)

---

## 6. Persistencia y Exportación
El conjunto transformado fue consolidado en un DataFrame estructurado y exportado a:
* **Ruta de destino:** `data/Customer_preprocessed.csv`
* **Estado:** Listo para entrenamiento o segmentación.

## 7. Como correr el archivo .py
python src/preprocess_customer.py