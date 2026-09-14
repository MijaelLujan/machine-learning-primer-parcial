# Informe Técnico: Preprocesamiento de Datos con Feature Engineering Territorial

**Dataset:** `Customer_clean.csv`

**Notebook:** `02_preprocesamiento_customerNicole.ipynb`

**Fase:** Ingeniería de Características y Preparación de Datos (Feature Engineering & Preprocessing)

---

## 1. Resumen Ejecutivo

Se actualizó el flujo de preparación del conjunto de datos de clientes (`Customer_clean.csv`, 793 registros) para incorporar información de localización fina sin incurrir en la maldición de la dimensionalidad. Mediante extracción por prefijo postal, estandarización numérica y vectorización binaria, se generó una matriz estructurada orientada a tareas de segmentación territorial y demográfica (K-Means, DBSCAN) o clasificación supervisada.

---

## 2. Diagnóstico y Tratamiento del Código Postal

* **Problema previo:** La columna original `postal_code` posee una cardinalidad elevada que, al aplicarse codificación One-Hot directa, generaría cientos de variables dispersas sobreajustando el volumen muestral (793 filas).


* **Técnica aplicada (Feature Extraction):** Se extrajo el primer carácter alfanumérico del código postal (`postal_code.astype(str).str[0]`) para definir la nueva característica `postal_zone`.
* **Beneficio analítico:** Reduce cientos de valores individuales a un rango acotado de zonas macrosectoriales, preservando la proximidad territorial relativa sin saturar el espacio vectorial.

---

## 3. Criterios de Selección y Descarte de Variables

* **Variables Descartadas (`remainder='drop'`):**
* `customer_id` y `customer_name`: Identificadores unívocos no generalizables.


* `postal_code` original: Sustituido funcionalmente por `postal_zone`.
* `city` y `state`: Omitidas para prevenir colinealidad geográfica y sobredispersión.




* **Variable Numérica Continua:**
* `age`: Variable demográfica central.




* **Variables Categóricas Nominales:**
* `segment`: Clasificación comercial (Consumer, Corporate, Home Office).


* `region`: División geográfica macro (Central, East, South, West).


* `postal_zone`: Macrozona postal calculada.



---

## 4. Transformaciones Aplicadas (`ColumnTransformer`)

### A. Estandarización Numérica (`StandardScaler`)

* **Variable:** `age`

* **Procedimiento:** Transformación Z-score Z= x-miu/sigma con media 0 y varianza unitaria.


* **Resultado:** Columna `num__age`, garantizando equilibrio en el cálculo de distancias euclidianas frente a variables binarias.



### B. Vectorización Categórica (`OneHotEncoder`)

* **Variables:** `segment`, `region`, `postal_zone`
* **Configuración:** `handle_unknown='ignore'`, `sparse_output=False`

* **Resultado:** Generación de columnas binarias (*dummy variables*):


* `cat__segment_*` (3 niveles)


* `cat__region_*` (4 niveles)


* `cat__postal_zone_*` (subconjunto representativo de dígitos postales)



---

## 5. Estructura de la Matriz Final

* **Dimensiones:** 793 filas x variables numéricas normalizadas/binarias.


* **Ausencia de nulos:** 100% de completitud sin valores faltantes.


* **Destino de exportación:** `data/Customer_preprocessed.csv`

* **Aptitud metodológica:** Matriz totalmente homogénea lista para modelos de agrupamiento espacial/demográfico o modelos de clasificación predictiva.