# Proyecto semestral 2, introducción a la inteligencia artificial.

Segundo proyecto semestral del curso Introducción a la intelligencia artificial.

## Descripción del problema

Dado un clip de audio musical, ¿es posible predecir su género (rock, jazz, clásica, hip-hop, etc.) usando únicamente su representación visual como espectrograma? Este proyecto convierte un problema de clasificación de audio en uno de clasificación de imágenes, donde cada clip se transforma en un espectrograma mel, y un modelo de visión pre-entrenado aprende a distinguir los patrones espectrales característicos de cada género.

Esta idea surge de una conversación que tuvimos respecto a música y como Spotify puede asociar canciones de distintos autores y distintas etiquetas para armar listas de reproducción. Y nos resultó en un problema interesante porque el género musical no depende solo de un instrumento o frecuencia puntual, sino de patrones rítmicos distribuidos en el tiempo — algo que un CNN, diseñado para detectar patrones espaciales locales, debería poder capturar razonablemente bien aunque nunca haya "escuchado" música en su preentrenamiento original si primero convertimos el audio en imágenes a través de los espectrogramas (usaremos ImageNet para esto).