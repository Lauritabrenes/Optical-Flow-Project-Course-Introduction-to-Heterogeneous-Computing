# Onboarding — Prototipo en C (Optical Flow: Kria vs. Orin Nano)

Este documento resume el panorama del proyecto, la estructura del repositorio, el flujo del código y la forma de trabajo esperada para las cuatro personas del equipo. El objetivo es que cualquiera pueda retomar el trabajo sin depender de explicaciones verbales.

## Panorama general del proyecto

El proyecto corresponde a la evaluación final de EL5859 Computación Heterogénea (TEC), a cargo del profesor Luis G. León-Vega. Se compara la aceleración del cálculo de optical flow (Dense Pyramidal Lucas-Kanade y TV-L1) entre dos plataformas embebidas heterogéneas, siguiendo la misma metodología de tres etapas en ambas: un baseline en C con optimizaciones NEON sobre el procesador ARM, seguido de una etapa de aceleración específica de cada plataforma.

Cambio importante respecto al planteamiento original: el profesor indicó que el Jetson Nano quedó obsoleto y se reemplaza por el Orin Nano. Por ahora la comparación sigue siendo Kria KV260 (FPGA) vs. Orin Nano (GPU embebida), sin necesidad de replantear el proyecto por completo; el alcance podría ajustarse más adelante conforme avance el curso. El acceso a la Orin Nano estará disponible en aproximadamente dos semanas, así que el prototipo en C se está desarrollando y probando en la Kria mientras tanto.

El curso pondera el proyecto en avances puntuales: estudio del estado del arte (5 %), diseño (5 %), prototipo en C (10 %, semana 7, con primer release), prototipo en GPU (10 %, semana 10), prototipo en FPGA (10 %, semana 14), análisis heterogéneo (5 %, semana 16), y defensa más artículo (15 %, semana 18). Actualmente estamos en la etapa de prototipo en C, con un plazo interno de tres días para el primer release.

## Estructura del repositorio y por qué existe cada carpeta

El repositorio no organiza las carpetas por etapa del algoritmo, sino por responsabilidad dentro del proyecto:

- `src/` — el código en C que hace el trabajo real: carga de imágenes, preprocesamiento, algoritmos de optical flow y cálculo de métricas.
- `include/of.h` — la interfaz compartida. Define las estructuras de datos (`OFImage`, `OFFlow`) y las firmas de todas las funciones. Es el contrato entre las cuatro etapas: nadie debe modificarlo sin acuerdo del equipo, porque un cambio ahí afecta a las cuatro personas a la vez.
- `data/` — los datasets e imágenes de entrada usados para pruebas (por ejemplo, secuencias de Middlebury o KITTI 2015).
- `results/` — lo que el programa produce al medir: tiempos, EPE, benchmarks por resolución.
- `scripts/` — herramientas auxiliares que no necesitan estar en C (por ejemplo, scripts de Python para graficar resultados).
- `third_party/` — código externo que se incorpore al proyecto (por ejemplo, referencias de la Vitis Vision Library si se usan como guía).
- `CMakeLists.txt` — le dice al sistema cómo compilar todos los `.c` de `src/` en un único ejecutable (`bench_main`).
- `toolchain-aarch64.cmake` — archivo opcional para cuando se necesite compilar en cruzado (cross-compile) desde una PC hacia arquitectura ARM aarch64.
- `.gitignore` — evita que se suban artefactos de compilación (`build/`, `*.o`, etc.) al repositorio.
- `README.md` — instrucciones generales para reproducir el proyecto.
- `ONBOARDING.md` (este archivo) — panorama del proyecto y forma de trabajo.

## Flujo del código: las cuatro etapas

El programa sigue un flujo secuencial de cuatro etapas, y cada una corresponde a uno o dos archivos en `src/`:

```
bench_main
   |
   v
1. Inicializacion y carga        -> io.c
   |
   v
2. Preprocesamiento              -> preprocess.c
   (uint8 -> float32, buffers)
   |
   v
3. Algoritmos de optical flow    -> lk_scalar.c / lk_neon.c
   (piramide, gradientes, LK)
   |
   v
4. Resultados y mediciones       -> metrics.c
   (EPE, tiempo, benchmarks)
   |
   v
results/
```

Es importante no confundir "preprocesamiento" con el algoritmo en sí: la conversión a float32 y la organización de buffers son preprocesamiento, pero la construcción de la pirámide y los gradientes ya son parte del algoritmo de optical flow (etapa 3). Todavía no sabemos cuál etapa será el cuello de botella; eso lo va a indicar el profiling una vez que el prototipo funcione de extremo a extremo.

## Cómo se organiza el trabajo en equipo

El equipo son cuatro personas. Por ahora los roles se identifican como Persona 1, 2, 3 y 4 (aún no se han asignado formalmente):

- Persona 1 — Inicialización y carga (`io.c`)
- Persona 2 — Preprocesamiento (`preprocess.c`)
- Persona 3 — Algoritmos de optical flow (`lk_scalar.c`, `lk_neon.c`)
- Persona 4 — Resultados y mediciones (`metrics.c`)

Flujo de trabajo en Git:

1. `main` ya contiene la estructura base, `include/of.h` acordado y un `CMakeLists.txt` que compila con stubs (funciones vacías con `TODO`). Cualquiera puede clonar y compilar desde ahora mismo.
2. Cada persona crea su propia rama desde `main`: `feature/io`, `feature/preprocess`, `feature/lk`, `feature/metrics`.
3. Cada quien trabaja únicamente en el o los archivos de su etapa, reemplazando el `TODO` por la implementación real. No se debe modificar `of.h` sin avisar al resto del equipo.
4. Se hacen commits y pull requests pequeños y frecuentes, en vez de un único PR grande al final.
5. Punto de integración: hacia la tarde del segundo día, las ramas se van fusionando a `main` en el orden del pipeline (io -> preprocess -> lk -> metrics), verificando en cada fusión que el proyecto sigue compilando y corriendo de extremo a extremo.
6. El tercer día se dedica a benchmarks en las tres resoluciones (VGA, 720p, 1080p), ajustes finales y preparación del release.

## Notas técnicas importantes

- NEON es específico de ARM: el código de `lk_neon.c` no va a compilar en una laptop x86 normal. La versión escalar (`lk_scalar.c`) se puede desarrollar y probar en cualquier máquina, pero la versión NEON debe compilarse y probarse directamente en la Kria (por SSH), o usando el toolchain de cross-compile si se prueba desde la PC.
- Al hacer `git fetch`/`git pull` del repositorio apareció una rama `Developers` además de `main`. Antes de fusionar cualquier cosa hacia o desde esa rama, conviene confirmar con el resto del equipo qué contiene y para qué se está usando, para evitar pisar trabajo que ya esté ahí.
- Aunque en esta etapa las métricas de `metrics.c` pueden ser simples (tiempo y EPE), el proyecto eventualmente necesita medir también throughput (FPS), potencia promedio, energía por frame y EPE sobre Middlebury/KITTI 2015. Vale la pena que la Persona 4 diseñe `of_save_results` pensando en que esos campos se van a necesitar más adelante, aunque no se implementen todos ahora.
- El código de esta etapa (prototipo en C) es la base sobre la que luego se construyen los prototipos de GPU y FPGA: mientras más ordenado y reproducible quede ahora, menos retrabajo habrá en las siguientes semanas.
