# ¿Las métricas financieras reconstruyen los sectores del S&P 500?

Proyecto de aula de Machine Learning (Ingeniería Financiera, Universidad de Medellín) que evalúa si las métricas financieras y de valoración de las empresas del S&P 500, **por sí solas**, contienen suficiente información para recuperar su sector económico según la clasificación **GICS** (11 sectores).

El problema se aborda desde dos ángulos:

1. **Parte 1 – Aprendizaje no supervisado:** se agrupan las empresas *sin* mostrarles el sector (K-means, clustering jerárquico aglomerativo y DBSCAN) y luego se comparan los grupos con los sectores reales.
2. **Parte 2 – Aprendizaje supervisado:** se entrena una red neuronal para predecir el sector y se prueba si la reducción de dimensión (SFFS, PCA y LDA) mejora la clasificación.

## Autores

- María José Salazar Jaramillo
- María Juliana Quintero Duque
- Santiago Villegas Restrepo

---

## Pregunta de investigación

> ¿Las métricas financieras y de valoración, por sí solas, contienen suficiente información para reconstruir los sectores económicos sin conocerlos de antemano?

La clasificación sectorial es clave en valoración por comparables, construcción de portafolios y gestión de riesgo. Sin embargo, el GICS clasifica por **actividad económica**, no por **perfil financiero**. Si los modelos no recuperan los sectores, los grupos que encuentren pueden servir como una segmentación alternativa (por ejemplo, *value* vs. *growth*, o defensivas vs. volátiles).

## Datos

| Atributo | Detalle |
|---|---|
| Nombre | S&P 500 Companies with Financial Information |
| Fuente | [Kaggle – Payton Fisher (2020)](https://www.kaggle.com/datasets/paytonfisher/sp-500-companies-with-financial-information), con datos de Yahoo Finance |
| Tamaño | 505 acciones × 14 columnas (algunas empresas tienen más de una clase de acción, p. ej. GOOGL y GOOG) |
| Tipo | Corte transversal (una sola fotografía, no serie de tiempo) |
| Variable objetivo | `Sector` (11 clases GICS, desbalanceadas: de 84 empresas en Consumer Discretionary a 3 en Telecommunication Services) |

El notebook descarga el dataset automáticamente con `kagglehub`.

### Variables de entrada finales (9)

`Price/Earnings`, `Dividend Yield`, `Earnings/Share`, `Market Cap`, `EBITDA`, `Price/Sales`, `Price/Book`, `Posicion_52s` y `Amplitud_52s`.

Las dos últimas son derivadas y reemplazan a `Price`, `52 Week Low` y `52 Week High`, que tenían una correlación de Spearman cercana a 1 entre sí y dependen del número de acciones emitidas:

- `Posicion_52s = (Price − Low) / (High − Low)` → ubicación del precio dentro del rango anual (*momentum*).
- `Amplitud_52s = (High − Low) / Price` → amplitud del rango respecto al precio (*proxy* de volatilidad).

## Metodología

### Limpieza y análisis exploratorio
- Se elimina `SEC Filings` (URL sin información). `Symbol` y `Name` se conservan solo para identificar empresas.
- **Corrección de datos:** las columnas `52 Week Low` y `52 Week High` venían intercambiadas en la fuente y se renombran.
- Imputación con la **mediana** de los 10 valores nulos (`Price/Earnings` y `Price/Book`), dado que las distribuciones son muy asimétricas.
- 33 empresas con precio ligeramente inferior a su mínimo de 52 semanas (máximo −2,7 %): se atribuye a un desfase de fechas y su posición se acota a [0, 1].

### Preprocesamiento
1. **Logaritmo con signo**, `sign(x)·ln(1+|x|)`, en las variables de cola pesada (admite valores negativos como EPS o EBITDA).
2. **Winsorización** en los percentiles 1 y 99.
3. **Estandarización** (`StandardScaler`).

En la Parte 2, los límites de winsorización y el escalador se ajustan **solo con el conjunto de entrenamiento** (75/25 estratificado: 378 / 127 empresas) para evitar fuga de información.

### Parte 1 – Clustering
| Modelo | Configuración final | Cómo se eligió |
|---|---|---|
| K-means | k = 11, `n_init=10` | Método del codo y silueta para k = 2…15 |
| Aglomerativo | Enlace *complete*, k = 11 | Comparación de enlaces *single*, *complete* y *centroid* (dendrogramas, correlación cofenética, tamaño de grupos) |
| DBSCAN | `eps = 2,0`, `min_samples = 10` | Gráfico de k-distancias y barrido de parámetros |

Los grupos se comparan con los sectores usando métricas **externas**: ARI, NMI, AMI, homogeneidad, completitud, pureza, exactitud con asignación húngara 1 a 1 y F1 por sector.

### Parte 2 – Red neuronal y reducción de dimensión
- **Selección de características con SFFS** (*Sequential Floating Forward Selection*, `mlxtend`) usando como *wrappers* regresión logística, árbol de decisión y Random Forest.
- **Red neuronal** (Keras): búsqueda de arquitectura variando épocas, número de capas ocultas y activación (ReLU vs. sigmoide), con validación cruzada estratificada de 5 *folds*. Arquitectura elegida: **32 → 16 neuronas, activación sigmoide, salida softmax de 11 clases, 200 épocas**, optimizador Adam.
- **PCA y LDA**: se prueban de 1 a 9 dimensiones con la misma red y se compara contra SFFS y contra usar todas las variables.

## Resultados principales

### Parte 1 – ¿Qué modelo de clustering se parece más a los sectores GICS?

| Métrica | K-means | Aglomerativo (complete) | DBSCAN |
|---|---|---|---|
| ARI | **0,157** | 0,061 | 0,043 |
| AMI | **0,246** | 0,156 | 0,151 |
| Exactitud (asignación 1 a 1) | **36,6 %** | 27,1 % | 22,2 % |
| F1 promedio por sector | **0,367** | 0,260 | 0,221 |
| Sectores con F1 > 0,5 | Real Estate, Financials | Ninguno | Financials |

Línea base (asignar todo al sector más grande): **16,6 %**.

- **K-means** es el mejor: recupera con claridad **Real Estate** (F1 = 0,83) y **Financials** (F1 = 0,77), y parcialmente Utilities y Energy.
- Los tres modelos agrupan más por **estilo de inversión** (defensivo, *growth*, *megacap*, empresas con pérdidas) que por industria.
- El grupo financiero se forma en buena parte porque la fuente asigna **EBITDA = 0** a bancos y aseguradoras.

### Parte 2 – Red neuronal (conjunto de prueba, 127 empresas)

| Variables de entrada | N.º variables | F1 macro CV | F1 macro prueba | Accuracy prueba |
|---|---|---|---|---|
| Todas las características | 9 | **0,414** ± 0,060 | 0,382 | 0,417 |
| SFFS + Regresión logística | 8 | 0,403 ± 0,058 | 0,362 | 0,402 |
| SFFS + Random Forest | 6 | 0,359 ± 0,069 | 0,383 | 0,425 |
| LDA | 8 | 0,410 ± 0,067 | 0,364 | 0,402 |
| PCA | 9 | 0,409 ± 0,067 | 0,354 | 0,386 |

- Las diferencias entre alternativas son menores que sus intervalos de confianza: **ninguna reducción mejora significativamente la red**, porque las variables redundantes ya se habían eliminado en la Parte 1.
- Los tres métodos SFFS coinciden en descartar `Price/Earnings`.
- **LDA** concentra la separación entre sectores en pocas dimensiones (94 % con 4 discriminantes); **PCA** necesita 6 componentes para explicar el 80 % de la varianza y, con pocas dimensiones, clasifica mucho peor (F1 de 0,18 vs. 0,34 de LDA con 2 dimensiones).
- La red reconoce bien Real Estate (recall 0,88), Financials y Consumer Staples (precisión 1,00), pero nunca predice Materials ni Telecommunication Services.

## Conclusión

Las métricas financieras **no contienen suficiente información** para reconstruir los 11 sectores GICS: el mejor modelo no supervisado ubica correctamente menos del 40 % de las empresas, y la mejor red neuronal acierta en 53 de 127 empresas de prueba. Solo se identifican bien los sectores con una huella financiera propia: **Inmobiliario** (dividendo alto, estructura REIT), **Financiero** (EBITDA = 0) y **Servicios públicos** (estabilidad).

Aun así, se confirma la hipótesis alternativa: las métricas permiten construir una **segmentación por perfil financiero** (clusters de K-means, ejes de PCA) útil en gestión de riesgo de portafolios —p. ej., separar empresas refugio de empresas volátiles— o para elegir comparables en una valoración según su perfil y no solo su sector.

## Estructura del notebook

| Sección | Contenido |
|---|---|
| Descripción de la base de datos | Fuente, dominio, pregunta de investigación, variables y tipo de problema |
| Análisis exploratorio | Limpieza, corrección de columnas, imputación, distribuciones, correlaciones, variables derivadas y perfil por sector |
| Preprocesamiento | Logaritmo con signo, winsorización y estandarización |
| Parte 1 | K-means, clustering aglomerativo, DBSCAN y comparación frente a los sectores GICS |
| Parte 2 | SFFS, red neuronal (búsqueda de arquitectura), PCA, LDA y comparación de técnicas de reducción |
| Conclusiones finales | Respuesta a la pregunta de investigación |

## Cómo ejecutarlo

El notebook fue desarrollado en **Google Colab**. Para correrlo localmente:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy keras tensorflow mlxtend kagglehub
jupyter notebook Proyecto_de_aula_final.ipynb
```

- `kagglehub` descarga el dataset automáticamente (puede requerir credenciales de Kaggle).
- Las semillas están fijadas (`random_state=42` y `keras.utils.set_random_seed`), pero los resultados de la red neuronal pueden variar ligeramente según la versión de TensorFlow/Keras y el hardware.
- La Parte 2 tarda varios minutos: cada configuración de la red se entrena con validación cruzada de 5 *folds* (≈ 30–140 s por configuración).

## Tecnologías

Python · pandas · NumPy · scikit-learn · SciPy · Keras/TensorFlow · mlxtend · Matplotlib · Seaborn · kagglehub
