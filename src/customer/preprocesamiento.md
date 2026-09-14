# Preprocesamiento de Datos: Clasificación Geográfica por Región

En esta etapa preparamos el dataset limpio (`Customer_clean.csv`) para abordar un problema de **aprendizaje supervisado (clasificación multiclase)**, donde el objetivo principal es predecir la macrozona territorial (`region`) a partir de variables de localización geográfica: `country`, `state`, `city` y `zip` (código postal).

---

## 1. Definición del Problema y Variables

* **Variable Objetivo ($y$):** `region` (etiqueta a predecir: Central, East, South, West).
* **Variables Predictoras ($X$):**
  * `country`: País de residencia.
  * `state`: Estado o provincia.
  * `city`: Ciudad.
  * `zip`: Código postal (estandarizado a partir de `postal_code`).
* **Variables omitidas:** Se dejaron fuera identificadores únicos (`customer_id`, `customer_name`) y variables demográficas no geográficas (`age`, `segment`) para enfocar el modelo puramente en la consistencia y jerarquía territorial.

---

## 2. Decisiones Técnicas y Transformaciones

* **Homogeneización del código postal:** Se detectó la columna original `postal_code` y se transformó como `zip` en formato texto (string) para evitar que los algoritmos traten los números postales como magnitudes continuas.
* **Vectorización con `OneHotEncoder`:** 
  * Se procesaron todas las características categóricas geográficas usando `ColumnTransformer`.
  * Se configuró con `sparse_output=False` para generar una matriz densa manipulable en pandas.
  * Se añadió `handle_unknown='ignore'` para que el pipeline ignore categorías o ciudades nuevas que puedan aparecer durante la inferencia o en el conjunto de prueba sin romper el código.
* **Estratificación (`train_test_split`):**
  * División 80% entrenamiento y 20% test con semilla fija (`random_state=42`).
  * Se aplicó `stratify=y` para asegurar que las proporciones relativas de cada clase en `region` se conserven idénticas en ambas particiones.

---

## 3. Salida Generada

* **Archivo final:** `data/Customer_region_preprocessed.csv`
* **Estructura:** Matriz de características totalmente codificadas en variables binarias (0/1) con nombres descriptivos obtenidos mediante `get_feature_names_out()`, anexando al final la columna de referencia `target_region`.
* **Uso posterior:** Listo para entrenar y evaluar modelos de clasificación multiclase (como Decision Trees, Random Forest o Regresión Logística Multinomial).