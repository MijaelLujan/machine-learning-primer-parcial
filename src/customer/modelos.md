# Clasificación de Clientes por Segmento

En esta etapa utilizamos el dataset limpio (`Customer_clean.csv`) para abordar un segundo problema de **aprendizaje supervisado (clasificación multiclase)**, complementario al de macrozona (`region`): predecir el tipo de cliente (`segment`) a partir de variables demográficas y geográficas.

---

## 1. Definición del Problema y Variables

* **Variable Objetivo ($y$):** `segment` — 3 clases: `Consumer` (409 registros, 51.6%), `Corporate` (236, 29.8%), `Home Office` (148, 18.7%).
* **Variables Predictoras ($X$):**
  * `age`: Edad del cliente (numérica).
  * `city`: Ciudad (252 valores únicos).
  * `state`: Estado o provincia (41 valores únicos).
  * `postal_code`: Código postal, tratado como texto para no interpretarlo como magnitud continua.
  * `region`: Macrozona territorial (Central/East/South/West), reutilizada como predictor aquí aunque en el otro problema es el propio target.
* **Variables omitidas:** `customer_id`, `customer_name` (identificadores sin señal predictiva generalizable).
* **Hipótesis de partida:** a diferencia de `region` (que depende directamente de `state`/`city` por definición geográfica), no hay una relación causal obvia entre ubicación/edad y el tipo de cliente — esa relación depende más de comportamiento de compra, que no está en este dataset. Se documenta esta hipótesis porque los resultados (sección 5) la confirman.

---

## 2. Preprocesamiento

* **Escalado de variables numéricas:** `StandardScaler` sobre `age`, necesario porque SVM es sensible a la escala de las features (a diferencia de árboles, que son invariantes a escalado monótono).
* **Codificación categórica:** `OneHotEncoder` sobre `city`, `state`, `postal_code`, `region`, vía `ColumnTransformer`.
  * `sparse_output=False` para trabajar con matriz densa en pandas.
  * `handle_unknown='ignore'` para tolerar categorías no vistas en test/inferencia sin romper el pipeline.
* **Split:** 80/20 con `random_state=42` y `stratify=y`, para conservar la proporción de clases de `segment` en ambas particiones (relevante dado el desbalance 52/30/19% mencionado arriba).
* **Salida:** `data/Customer_segment_preprocessed.csv` — matriz codificada (546+ columnas por el one-hot de ciudad/estado/zip) más la columna `target_segment`.
* **Código:** `src/customer/preprocesamiento.py`, función `preprocess_segment_classification`. Convive en el mismo archivo con `preprocess_region_classification` (de `region`), sin compartir lógica entre ambas — cada una mantiene sus propias decisiones de features y transformación.

---

## 3. Modelos Entrenados

Mismo split y mismo preprocesamiento para los 3, variando solo el algoritmo — así la comparación aísla el efecto del modelo:

| Modelo | Implementación | Configuración |
|---|---|---|
| Regresión Logística | `sklearn.linear_model.LogisticRegression` | `max_iter=1000` (default no converge con este número de features tras one-hot) |
| SVM | `sklearn.svm.SVC` | kernel RBF por defecto |
| Árbol de Decisión | `sklearn.tree.DecisionTreeClassifier` | `random_state=42`, sin poda (`max_depth` sin restringir) |

**Estructura de código:** cada modelo vive en su propio módulo bajo `src/customer/modelos/` (`logistica.py`, `svm.py`, `arbol.py`), cada uno exponiendo `crear_modelo()` y una constante `NOMBRE`. `src/customer/modelos.py` orquesta: carga el CSV preprocesado, arma el split, itera sobre los 3 módulos entrenando y evaluando con el mismo criterio, e imprime una comparación final ordenada por accuracy. Esto permite agregar o quitar un modelo sin tocar el script de comparación.

---

## 4. Resultados

Accuracy sobre el 20% de test (159 registros):

| Modelo | Accuracy | Recall Consumer | Recall Corporate | Recall Home Office |
|---|---|---|---|---|
| SVM | 0.4843 | 0.90 | 0.00 | 0.10 |
| Regresión Logística | 0.4340 | 0.67 | 0.21 | 0.13 |
| Árbol de Decisión | 0.3396 | 0.48 | 0.17 | 0.23 |

---
