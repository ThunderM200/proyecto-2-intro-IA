# Proyecto semestral 2, introducción a la inteligencia artificial.

Segundo proyecto semestral del curso Introducción a la intelligencia artificial.

## Descripción del problema

Dado un clip de audio musical, ¿es posible predecir su género (rock, jazz, clásica, hip-hop, etc.) usando únicamente su representación visual como espectrograma? Este proyecto convierte un problema de clasificación de audio en uno de clasificación de imágenes, donde cada clip se transforma en un espectrograma mel, y un modelo de visión pre-entrenado aprende a distinguir los patrones espectrales característicos de cada género.

Esta idea surge de una conversación que tuvimos respecto a música y como Spotify puede asociar canciones de distintos autores y distintas etiquetas para armar listas de reproducción. Y nos resultó en un problema interesante porque el género musical no depende solo de un instrumento o frecuencia puntual, sino de patrones rítmicos distribuidos en el tiempo — algo que un CNN, diseñado para detectar patrones espaciales locales, debería poder capturar razonablemente bien aunque nunca haya "escuchado" música en su preentrenamiento original si primero convertimos el audio en imágenes a través de los espectrogramas (usaremos ImageNet para esto).

## Dataset

**GTZAN Genre Collection** (Kaggle: [`andradaolteanu/gtzan-dataset-music-genre-classification`](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)).

- 1000 clips de audio de 30 segundos (formato `.wav`).
- 10 géneros, 100 clips cada uno: `blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock`.
- Dataset balanceado por diseño.
- Fuente original: G. Tzanetakis y P. Cook, *"Musical Genre Classification of Audio Signals"* (2002).

Nosotros tomamos clips de 3 segundos para convertirlos a espectrogramas, lo que nos ayuda a reducir el tamaño de los elementos individuales del entrenamiento y lleva el dataset a ser 10 veces más grande que lo que tenemos de base.

## Justificación del modelo

Se utiliza **transfer learning con ResNet18 pre-entrenado en ImageNet** (`torchvision.models`), en lugar de entrenar un CNN desde cero para la realización de este proyecto.

**Ventajas de usar el modelo:**

- El dataset es pequeño (~800 canciones para entrenamiento tras el split, lo que nos resulta en ~8000 segmentos para entrenamiento, que dada la correlación existente entre ellos nos deja con menos elementos efectivos para el entrenamiento). Y si entrenaramos de cero una CNN, llegaríamos rápido a problemas por sobre ajuste del modelo.

- Los filtros de bajo nivel aprendidos en ImageNet (bordes, texturas, gradientes) son razonablemente transferibles a espectrogramas, que también son imágenes con estructura local (bandas de energía, patrones armónicos).

- Permite converger en pocas épocas y con un costo computacional relativamente bajo.

**Limitaciones:**

- ResNet18 fue entrenado en imágenes de fotografías naturales, no en espectrogramas, por lo que existe una diferencia en el dominio del modelo y no se puede garantizar el mismo nivel de categorización y generalización que en tareas de clasificación de fotografías.

- Perder el contexto temporal de la canción completa. Como usamos segmentos pequeños sobre un dataset que también es relativamente pequeño, la capacidad de entrenar el modelo es limitada.

**Estrategia:** Entrenamiento de dos etapas, donde primero se congela el backbone y se entrena solo las primeras capas del modelo, luego se descongelan las capas convolucionales finales (`layer4`) con un *learning rate* bajo para que la categorización sea más precisa.

## Metodología (paso a paso)

1. **Carga de datos:** se indexan los archivos `.wav` por género en un DataFrame.

2. **EDA:** verificación de balance de clases, inspección de forma del dataset y su composición del dataset.

3. **Feature engineering:** cada clip se segmenta en ventanas de 3s y se convierte a espectrograma mel en escala logarítmica (dB) con `librosa`. Los espectrogramas se cachean como `.npy`.

4. **Split:** división train/val/test estratificada por género y agrupada por canción original para evitar *data leakage* entre segmentos de la misma canción.

5. **Dataset y DataLoader:** los espectrogramas se normalizan, se replican a 3 canales, se redimensionan a 224x224 y se normalizan con las estadísticas de ImageNet. En entrenamiento se aplica **SpecAugment** (enmascaramiento de frecuencia y tiempo) como forma de aumentar los datos con los que trabajamos.

6. **Modelo:** ResNet18 pre-entrenado, cabeza reemplazada por `Dropout + Linear(10)`.

7. **Entrenamiento:**
   
   - Fase 1: backbone congelado, solo se entrena la cabeza del modelo.
   
   - Fase 2: fine-tuning de `layer4` con LR reducido para mejorar la categorización del modelo.

   - Regularización: dropout, weight decay (L2), data augmentation, early stopping por `val_loss`, `ReduceLROnPlateau` como scheduler para el entrenamiento.

8. **Testeo:** evaluación en el set de test con accuracy, F1-score, classification report y matriz de confusión.

9. **Visualización:** curvas de entrenamiento, matriz de confusión, ejemplos de espectrogramas mal clasificados.

## Resultados obtenidos

El modelo alcanzó una **accuracy de 55%** y un **F1-score de 0.54** sobre el conjunto de prueba, superando el **baseline aleatorio de 10%** esperado para un problema de clasificación balanceado con diez géneros musicales, usando el entrenamiento solo hasta la fase 1. Estos resultados indican que la red fue capaz de aprender representaciones útiles de los espectrogramas, aunque todavía existe un margen importante de mejora para diferenciar géneros con características acústicas similares, que se esperan alcanzar al ejecutar la segunda fase de entrenamiento.

Una vez ejecutada la fase 2 del entrenamiento del modelo, se alcanzó un **accuracy del 73%** y un **F1-score de 0.73**, Que es un salto significativo sobre el entrenamiento inicial que incluía solo hasta la fase 1 del entrenamiento.

El análisis de la matriz de confusión muestra un desempeño relativamente homogéneo entre la mayoría de las clases. Los mejores resultados se obtuvieron para classical (190 aciertos de 200), jazz (179), metal (167), country (162), hiphop (158) y pop (157), mientras que rock continúa siendo la clase con menor tasa de aciertos (72 de 200). Las confusiones más frecuentes se producen entre rock y country (48 ejemplos clasificados como country), disco y pop (32), disco e hiphop (25), reggae y pop (25) y rock y pop (24). Estas confusiones resultan razonables desde el punto de vista musical, ya que dichos géneros comparten patrones rítmicos, instrumentación y características de sonido que pueden generar espectrogramas similares. 


## Conclusiones

Finalmente, el enfoque presenta varias limitaciones. En primer lugar, **GTZAN** es un conjunto de datos relativamente pequeño para entrenar modelos de deeep learning, por lo que la capacidad de generalización es limitada. 

Además, trabajar con segmentos de **3 segundos** implica perder parte significativa del contexto musical de la canción, como la estructura completa o la evolución temporal de sus patrones rítmicos y armónicos que se hubieran convertido a espectrograma. 

Además, buscando referencias acerca del dataset encontramos que hay documentos que dicen de la existencia de duplicados y datos imprecisos en GTZAN, lo que introduce una fuente adicional de error tanto durante el entrenamiento como en la evaluación del modelo.

Referencia: Sturm, B. L. (2013). The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use. arXiv:1306.1461.


## Estructura del repositorio

```
.
├── README.md
├── environment.yml
├── Codigo_Proyecto_Arnaldo_Gonzalez_Martin_Sandoval.ipynb   # notebook principal (EDA -> modelo -> resultados)
├── data/                                    # dataset GTZAN 
└── spectrograms/                            # espectrogramas cacheados (generados por el notebook, no versionado)
```

## Cómo correrlo

1. Crear el entorno:
   ```bash
   conda env create -f environment.yml
   conda activate proyecto-2-intro-ia
   ```
2. Descargar el dataset (requiere cuenta de Kaggle y `kaggle.json` en `~/.kaggle/`):

   - Esto se puede hacer a través del jupyter notebook en las celdas iniciales a través de kagglehub.

3. Confirmar que los audios quedaron en `data/genres_original/<genero>/*.wav`.

4. Abrir y ejecutar `Codigo_Proyecto_Arnaldo_Gonzalez_Martin_Sandoval.ipynb` de principio a fin.
