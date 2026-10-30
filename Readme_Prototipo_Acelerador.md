# Optical Flow — Prototipo en C

Proyecto final del curso **EL5859 — Introducción a la Computación Heterogénea**  
Instituto Tecnológico de Costa Rica

**Profesor:** Luis G. León Vega, Ph.D.

**Integrantes:**
- George Roussel Briceño Celestino
- Laura Elena Brenes Espinoza
- Marlon Méndez Naranjo
- Rodrigo Venegas Mora

---

## 1. Descripción

Este proyecto implementa un sistema de **estimación de movimiento mediante Optical Flow**, utilizando el método de **Lucas–Kanade**.

La finalidad del **Prototipo en C** es contar con una implementación funcional en CPU que permita:

- comprobar el funcionamiento completo del algoritmo;
- medir el costo de cada etapa;
- obtener una referencia de rendimiento;
- identificar los principales cuellos de botella;
- aplicar una primera optimización mediante ARM NEON;
- utilizar posteriormente estos resultados como baseline para las etapas de aceleración en GPU y FPGA.

El pipeline implementado es:

```text
Frame 1 + Frame 2
       |
       v
┌─────────────────────────────┐
│ 1. Inicialización y carga   │
│          src/io.c           │
└─────────────┬───────────────┘
              |
              v
┌─────────────────────────────┐
│ 2. Preprocesamiento         │
│      src/preprocess.c       │
└─────────────┬───────────────┘
              |
              v
┌─────────────────────────────┐
│ 3. Lucas–Kanade             │
│      src/lk_scalar.c        │
│      src/lk_neon.c          │
└─────────────┬───────────────┘
              |
              v
┌─────────────────────────────┐
│ 4. Métricas y resultados   │
│       src/metrics.c         │
└─────────────────────────────┘
```

---

# 2. Ruta rápida para revisar el Prototipo en C

Para facilitar la evaluación, los archivos principales pueden revisarse en este orden:

| Orden | Archivo / carpeta | Contenido |
|---:|---|---|
| 1 | `README.md` | Guía general, compilación, ejecución y resultados |
| 2 | `docs/decisions.md` | Decisiones técnicas y contratos entre etapas |
| 3 | `include/of.h` | Interfaz compartida del pipeline |
| 4 | `src/main.c` | Integración de todas las etapas y medición de tiempos |
| 5 | `src/io.c` | Inicialización, carga y conversión a escala de grises |
| 6 | `src/preprocess.c` | Normalización y filtro Gaussiano |
| 7 | `src/lk_scalar.c` | Lucas–Kanade escalar utilizado como baseline |
| 8 | `src/lk_neon.c` | Optimización SIMD ARM NEON |
| 9 | `src/metrics.c` | EPE y almacenamiento de resultados |
| 10 | `tests/` | Pruebas funcionales |
| 11 | `scripts/profile.sh` | Automatización del perfilado |
| 12 | `scripts/analyze.py` | Análisis estadístico |
| 13 | `results/profiling/` | Datos crudos de las corridas |
| 14 | `Video etapa Prototipo en C.mp4` | Demostración del prototipo |

---

# 3. Organización del repositorio

```text
.
├── CMakeLists.txt
├── README.md
├── ONBOARDING.md
│
├── include/
│   └── of.h
│
├── src/
│   ├── main.c
│   ├── io.c
│   ├── preprocess.c
│   ├── lk_scalar.c
│   ├── lk_neon.c
│   ├── metrics.c
│   │
│   ├── lk_cuda.cu
│   ├── lk_cuda_opt.cu
│   ├── lk_cuda_shared.cu
│   └── gpu_timing.cu
│
├── tests/
│   ├── test_io.c
│   ├── test_preprocess.c
│   ├── test_metrics.c
│   └── test_cuda_correctness.c
│
├── data/
│   ├── frame1.ppm
│   ├── frame2.ppm
│   └── frames/
│       ├── frame_000001.png
│       ├── frame_000002.png
│       ├── frame_0001.flo
│       └── frame_0002.flo
│
├── docs/
│   ├── decisions.md
│   └── Marlon_ai_interaction_log.md
│
├── scripts/
│   ├── profile.sh
│   ├── analyze.py
│   └── profile_gpu.sh
│
├── results/
│   ├── last_run.csv
│   └── profiling/
│       └── times.csv
│
├── third_party/
│   └── stb_image.h
│
├── toolchain-aarch64.cmake
└── Video etapa Prototipo en C.mp4
```

## ¿Qué contiene cada carpeta?

### `include/`

Contiene la interfaz común del proyecto.

El archivo principal es:

```text
include/of.h
```

Aquí se encuentran las estructuras:

```c
OFImage
OFFlow
```

y las funciones públicas utilizadas para conectar las diferentes etapas.

`of.h` funciona como **contrato común del equipo** y no se modifica sin coordinación entre los integrantes.

---

### `src/`

Contiene la implementación del algoritmo.

Para el **Prototipo en C** los archivos principales son:

```text
src/io.c
src/preprocess.c
src/lk_scalar.c
src/lk_neon.c
src/metrics.c
src/main.c
```

Los archivos:

```text
src/lk_cuda.cu
src/lk_cuda_opt.cu
src/lk_cuda_shared.cu
src/gpu_timing.cu
```

corresponden a etapas posteriores de aceleración mediante GPU y no son necesarios para comprender el baseline del Prototipo en C.

---

### `tests/`

Contiene pruebas diseñadas para verificar componentes individuales del proyecto.

```text
test_io.c
```

Comprueba la carga de imágenes y la conversión RGB → escala de grises.

```text
test_preprocess.c
```

Comprueba la normalización de intensidades y el comportamiento del filtro Gaussiano.

```text
test_metrics.c
```

Comprueba el cálculo del End-Point Error y la escritura de resultados.

`test_cuda_correctness.c` pertenece a la validación posterior de las implementaciones GPU.

---

### `data/`

Contiene datos utilizados para ejecutar y validar el prototipo.

```text
data/frame1.ppm
data/frame2.ppm
```

son imágenes sintéticas pequeñas de **64 × 64 píxeles**, utilizadas para pruebas y perfilado base.

Dentro de:

```text
data/frames/
```

se encuentran dos imágenes reales:

```text
frame_000001.png
frame_000002.png
```

con resolución:

```text
1024 × 436 píxeles
```

Estas imágenes se utilizan para ejecutar el algoritmo sobre un caso más representativo.

---

### `scripts/`

Incluye herramientas para automatizar la toma y análisis de resultados.

```text
scripts/profile.sh
```

ejecuta repetidamente el programa y almacena los tiempos de cada etapa.

```text
scripts/analyze.py
```

calcula estadísticas sobre los tiempos obtenidos:

- media;
- mediana;
- desviación estándar;
- mínimo;
- máximo.

---

### `results/`

Contiene los resultados generados durante las ejecuciones.

```text
results/last_run.csv
```

guarda el campo de flujo de una ejecución.

```text
results/profiling/times.csv
```

contiene los tiempos crudos del perfilado base del Prototipo en C.

---

### `third_party/`

Contiene dependencias externas utilizadas por el proyecto.

Actualmente incluye:

```text
third_party/stb_image.h
```

utilizada para cargar imágenes desde C.

---

# 4. Interfaces principales

La representación de una imagen se define en `include/of.h`:

```c
typedef struct {
    float *data;
    int width;
    int height;
} OFImage;
```

El campo de flujo óptico se representa mediante:

```c
typedef struct {
    float *u;
    float *v;
    int width;
    int height;
} OFFlow;
```

Las funciones principales del Prototipo en C son:

```c
int of_load_frame(const char *path, OFImage *out);

int of_preprocess(
    const OFImage *in,
    OFImage *out
);

int of_lk_scalar(
    const OFImage *frame1,
    const OFImage *frame2,
    OFFlow *flow
);

int of_lk_neon(
    const OFImage *frame1,
    const OFImage *frame2,
    OFFlow *flow
);

float of_compute_epe(
    const OFFlow *estimated,
    const OFFlow *ground_truth
);

void of_save_results(
    const char *path,
    const OFFlow *flow,
    double elapsed_ms
);
```

---

# 5. Etapa 1 — Inicialización y carga

**Archivo:**

```text
src/io.c
```

La función:

```c
of_load_frame()
```

recibe la ruta de una imagen.

Ejemplo:

```text
data/frames/frame_000001.png
```

La imagen es cargada mediante `stb_image`.

La decodificación se fuerza a RGB y posteriormente se convierte a escala de grises mediante la ecuación BT.601:

```text
Y = 0.299R + 0.587G + 0.114B
```

La salida de esta etapa es:

```text
OFImage
```

con:

```text
data   -> float32 en rango [0,255]
width  -> ancho de la imagen
height -> alto de la imagen
```

El buffer `data` se almacena de forma densa:

```text
width × height floats
```

sin stride ni padding adicional.

La normalización **no se realiza en esta etapa**, ya que pertenece a `preprocess.c`.

---

# 6. Etapa 2 — Preprocesamiento

**Archivo:**

```text
src/preprocess.c
```

La entrada es una imagen en escala de grises con valores:

```text
[0,255]
```

El preprocesamiento realiza primero:

```text
[0,255] → [0,1]
```

Posteriormente aplica un filtro Gaussiano separable con:

```text
sigma = 1.0
```

El filtro se ejecuta en dos pasos:

```text
horizontal
   ↓
vertical
```

Esto evita realizar directamente una convolución bidimensional completa.

La salida mantiene las mismas dimensiones de la imagen original.

---

# 7. Etapa 3 — Lucas–Kanade escalar

**Archivo:**

```text
src/lk_scalar.c
```

La implementación escalar funciona como **baseline funcional y de rendimiento**.

Para cada píxel se calculan los gradientes:

```text
Ix
Iy
It
```

donde:

```text
Ix = gradiente horizontal
Iy = gradiente vertical
It = diferencia temporal entre los dos frames
```

Los gradientes espaciales se calculan mediante diferencias centrales.

La diferencia temporal se calcula mediante:

```text
It = frame2 - frame1
```

El algoritmo utiliza una ventana local:

```text
7 × 7
```

equivalente a:

```text
LK_RADIUS = 3
```

Dentro de cada ventana se acumulan:

```text
Σ Ix²
Σ Iy²
Σ IxIy
Σ IxIt
Σ IyIt
```

y posteriormente se resuelve el sistema Lucas–Kanade de `2 × 2`.

Para evitar soluciones numéricamente inestables se utiliza:

```text
LK_DET_THRESHOLD = 1e-6
```

Los píxeles de borde y los puntos donde el sistema no puede resolverse mantienen flujo:

```text
u = 0
v = 0
```

---

# 8. Optimización ARM NEON

El Prototipo en C contempla una variante optimizada mediante **ARM NEON**.

El propósito de esta implementación es explotar paralelismo SIMD en procesadores ARM.

Mientras la versión escalar procesa datos utilizando instrucciones escalares, NEON dispone de registros vectoriales de 128 bits y permite trabajar con varios valores `float32` dentro de una misma operación SIMD.

La estrategia de optimización se concentra principalmente en el núcleo Lucas–Kanade, ya que el perfilado identifica esta etapa como la región con mayor costo computacional.

La versión escalar se conserva como referencia para verificar que la optimización no modifique el comportamiento esperado del algoritmo.

---

# 9. Etapa 4 — Métricas y resultados

**Archivo:**

```text
src/metrics.c
```

Esta etapa contiene dos funciones principales.

## End-Point Error

```c
of_compute_epe()
```

calcula el error promedio entre un campo estimado y un ground truth:

```text
EPE = sqrt((u_est - u_gt)² + (v_est - v_gt)²)
```

Cuando el ground truth contiene valores inválidos (`NaN` o infinito), estos puntos son omitidos.

---

## Almacenamiento de resultados

```c
of_save_results()
```

genera un archivo CSV con:

```text
width
height
elapsed_ms
u
v
```

Esto permite conservar tanto el tiempo de ejecución como el campo de flujo calculado.

---

# 10. Requisitos de software

El proyecto fue desarrollado para entornos Linux.

En Ubuntu/Debian pueden instalarse las herramientas básicas mediante:

```bash
sudo apt update
sudo apt install -y build-essential cmake python3
```

La dependencia:

```text
stb_image.h
```

ya se encuentra incluida dentro del repositorio, por lo que no requiere una instalación externa.

---

# 11. Compilación

Desde la raíz del proyecto:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

El proyecto utiliza C11.

En sistemas ARM/AArch64, la configuración contempla optimizaciones de compilación como:

```text
-O3
-march=armv8-a+simd
```

para habilitar las capacidades SIMD disponibles en la plataforma.

---

# 12. Ejecución

El programa principal recibe dos imágenes consecutivas.

Formato:

```bash
./build/bench_main <frame1> <frame2> [salida.csv] [--gt-shift DX DY]
```

Ejemplo con los frames reales incluidos:

```bash
./build/bench_main \
  data/frames/frame_000001.png \
  data/frames/frame_000002.png \
  results/last_run.csv
```

También puede ejecutarse desde la carpeta `build/`:

```bash
cd build

./bench_main \
  ../data/frames/frame_000001.png \
  ../data/frames/frame_000002.png \
  ../results/last_run.csv
```

---

# 13. Ground truth sintético

Para pares sintéticos donde el desplazamiento real es conocido se puede utilizar:

```text
--gt-shift DX DY
```

Ejemplo:

```bash
./build/bench_main \
  data/frame1.ppm \
  data/frame2.ppm \
  results/last_run.csv \
  --gt-shift 1.0 0.5
```

Esto crea un campo uniforme de referencia y permite calcular el EPE.

Este parámetro debe utilizarse únicamente cuando el desplazamiento verdadero del par de imágenes sea conocido.

---

# 14. Pruebas

## 14.1 Inicialización y carga

El archivo:

```text
tests/test_io.c
```

valida:

- carga correcta de la imagen;
- dimensiones;
- reserva de memoria;
- conversión RGB → escala de grises;
- ecuación BT.601.

La imagen:

```text
test.ppm
```

es una imagen controlada de `2 × 2` píxeles.

Ejecución:

```bash
./build/test_io test.ppm
```

Los valores esperados son:

```text
Rojo    -> 76.245
Verde   -> 149.685
Azul    -> 29.070
Blanco  -> 255.000
```

---

## 14.2 Preprocesamiento

El archivo:

```text
tests/test_preprocess.c
```

comprueba:

- validación de argumentos;
- normalización `[0,255] → [0,1]`;
- conservación de dimensiones;
- comportamiento del filtro Gaussiano;
- dispersión de un impulso central.

También puede compilarse directamente con GCC:

```bash
gcc -Iinclude \
  -o test_preprocess \
  src/preprocess.c \
  tests/test_preprocess.c \
  -lm
```

Ejecutar:

```bash
./test_preprocess
```

---

## 14.3 Métricas

El archivo:

```text
tests/test_metrics.c
```

comprueba:

- EPE igual a cero para flujos idénticos;
- caso conocido `(3,4) → EPE = 5`;
- validación de argumentos inválidos;
- generación del archivo CSV.

Cuando el target se encuentra habilitado en CMake:

```bash
./build/test_metrics
```

---

# 15. Perfilado

El script:

```text
scripts/profile.sh
```

ejecuta el programa varias veces y almacena los tiempos de cada etapa.

Formato:

```bash
bash scripts/profile.sh \
  <binario> \
  <frame1> \
  <frame2> \
  <numero_de_corridas> \
  <archivo_salida>
```

Ejemplo con 100 corridas:

```bash
bash scripts/profile.sh \
  build/bench_main \
  data/frame1.ppm \
  data/frame2.ppm \
  100 \
  results/profiling/times.csv
```

El CSV generado contiene:

```text
run
io_ms
preprocess_ms
lk_scalar_ms
save_ms
total_ms
```

---

# 16. Análisis estadístico

Después de generar los resultados:

```bash
python3 scripts/analyze.py results/profiling/times.csv
```

El script calcula:

```text
Media
Mediana
Desviación estándar
Mínimo
Máximo
```

para cada etapa del pipeline.

---

# 17. Resultados del perfilado base

El archivo:

```text
results/profiling/times.csv
```

incluye **100 corridas válidas** realizadas con el par sintético:

```text
data/frame1.ppm
data/frame2.ppm
```

Estas imágenes tienen resolución:

```text
64 × 64
```

Las estadísticas obtenidas son:

| Etapa | Media (ms) | Mediana (ms) | Desv. estándar (ms) | Mínimo (ms) | Máximo (ms) |
|---|---:|---:|---:|---:|---:|
| I/O | 0.122 | 0.135 | 0.041 | 0.067 | 0.211 |
| Preprocesamiento | 0.440 | 0.357 | 0.142 | 0.325 | 0.843 |
| Lucas–Kanade escalar | 21.406 | 21.093 | 1.134 | 20.306 | 29.917 |
| Guardado | 1.267 | 1.220 | 0.191 | 1.138 | 2.714 |
| **Total** | **23.235** | **22.972** | **1.203** | **22.002** | **32.036** |

El perfilado permite observar que la etapa de **Lucas–Kanade domina el tiempo de cómputo**, por lo que constituye la principal candidata para optimización y descarga hacia aceleradores.

---

# 18. Resultado ARM NEON documentado por el equipo

Para el par de imágenes reales incluido en:

```text
data/frames/
```

con resolución:

```text
1024 × 436
```

el benchmark documentado durante el desarrollo reportó:

| Implementación LK | Tiempo (ms) |
|---|---:|
| Escalar | 365.93 |
| ARM NEON | 147.57 |

El speedup correspondiente es:

```text
365.93 / 147.57 ≈ 2.48×
```

Esto representa una reducción aproximada del:

```text
59.7 %
```

en el tiempo del núcleo Lucas–Kanade.

Estos resultados corresponden específicamente al núcleo de cómputo LK y no deben confundirse con el tiempo total del pipeline.

---

# 19. Optimizaciones consideradas en el Prototipo en C

Durante esta etapa se consideraron principalmente:

### Implementadas

```text
Baseline escalar en C
Compilación Release
Optimización -O3
SIMD ARM NEON
Filtro Gaussiano separable
Buffers float32 contiguos
Perfilado individual por etapa
```

### No aplicadas dentro del Prototipo en C

Las siguientes optimizaciones corresponden a etapas posteriores:

```text
CUDA / GPU
Memoria compartida CUDA
FPGA / HLS
Procesamiento heterogéneo CPU + GPU + FPGA
```

Estas implementaciones pueden encontrarse también en el repositorio debido a la evolución posterior del proyecto, pero no forman parte del baseline del **Prototipo en C**.

---

# 20. Resolución de trabajo

En las decisiones iniciales del equipo se estableció como resolución objetivo:

```text
1280 × 720
```

Sin embargo, el programa permite ejecutar otras dimensiones siempre que ambos frames tengan el mismo tamaño.

Los frames reales actualmente incluidos en el repositorio son:

```text
1024 × 436
```

Por esta razón, los resultados deben reportarse siempre indicando la resolución utilizada en cada experimento.

---

# 21. Manejo de memoria

Las responsabilidades de memoria del pipeline son:

### `of_load_frame`

La función reserva:

```text
out->data
```

El llamador debe liberarlo posteriormente mediante:

```c
free()
```

### `of_preprocess`

El llamador reserva:

```text
out->data
```

La función solamente escribe sobre ese buffer.

Internamente utiliza un buffer temporal para el filtro Gaussiano, el cual es reservado y liberado dentro de la propia función.

### `of_lk_scalar` / `of_lk_neon`

El llamador reserva:

```text
flow->u
flow->v
```

Las funciones solamente escriben los resultados.

---

# 22. Salida del programa

Durante la ejecución, el programa reporta tiempos separados para las etapas principales:

```text
IO
PREPROCESS
LK_SCALAR
SAVE
TOTAL
```

Este formato es utilizado por:

```text
scripts/profile.sh
```

para construir automáticamente el archivo CSV de perfilado.

Además se muestran estadísticas básicas del campo de flujo, como:

```text
cantidad de píxeles con solución
promedio de u
promedio de v
magnitud máxima
```

---

# 23. Reproducibilidad

Para reproducir un experimento del Prototipo en C se recomienda registrar:

```text
Hardware utilizado
Sistema operativo
Arquitectura
Compilador
Versión de CMake
Resolución de entrada
Número de corridas
Tipo de implementación
Flags de compilación
```

El proyecto del curso solicita utilizar **más de 100 muestras por etapa** y realizar el perfilado en:

```text
al menos dos computadoras personales
+
un sistema empotrado
```

Por esta razón, los archivos CSV generados para cada plataforma deben conservarse dentro de `results/` o adjuntarse con la entrega correspondiente.

---

# 24. Video demostrativo

La demostración correspondiente al Prototipo en C se encuentra en:

```text
Video etapa Prototipo en C.mp4
```

Este video permite observar la compilación, ejecución y funcionamiento general del prototipo.

---

# 25. Relación con las etapas siguientes

El Prototipo en C funciona como referencia para las etapas posteriores del proyecto.

```text
Prototipo en C
     |
     v
Identificación del cuello de botella
     |
     v
ARM NEON
     |
     v
GPU / CUDA
     |
     v
FPGA / HLS
     |
     v
Análisis heterogéneo
```

El perfilado muestra que el cálculo de Lucas–Kanade representa la parte más costosa del pipeline.

Por esta razón, las etapas siguientes concentran la aceleración principalmente en esta sección, manteniendo la implementación escalar como referencia para comparar:

```text
correctitud
rendimiento
speedup
latencia
```

---

# 26. Resumen de comandos

## Instalar dependencias

```bash
sudo apt update
sudo apt install -y build-essential cmake python3
```

## Configurar

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

## Compilar

```bash
cmake --build build -j$(nproc)
```

## Ejecutar el Prototipo en C

```bash
./build/bench_main \
  data/frames/frame_000001.png \
  data/frames/frame_000002.png \
  results/last_run.csv
```

## Probar carga de imágenes

```bash
./build/test_io test.ppm
```

## Probar métricas

```bash
./build/test_metrics
```

## Probar preprocesamiento directamente

```bash
gcc -Iinclude \
  -o test_preprocess \
  src/preprocess.c \
  tests/test_preprocess.c \
  -lm

./test_preprocess
```

## Ejecutar 100 corridas

```bash
bash scripts/profile.sh \
  build/bench_main \
  data/frame1.ppm \
  data/frame2.ppm \
  100 \
  results/profiling/times.csv
```

## Analizar resultados

```bash
python3 scripts/analyze.py \
  results/profiling/times.csv
```

---

# 27. Archivos principales del entregable

Para la etapa **Prototipo en C**, los elementos principales son:

```text
README.md
include/of.h

src/main.c
src/io.c
src/preprocess.c
src/lk_scalar.c
src/lk_neon.c
src/metrics.c

tests/test_io.c
tests/test_preprocess.c
tests/test_metrics.c

scripts/profile.sh
scripts/analyze.py

results/profiling/times.csv

docs/decisions.md

Video etapa Prototipo en C.mp4
```

Los archivos relacionados con CUDA pertenecen a una etapa posterior y se mantienen dentro del mismo repositorio únicamente porque el proyecto continúa evolucionando sobre la misma base de código.

---

# 28. Conclusión

El Prototipo en C establece una implementación funcional y modular del algoritmo Lucas–Kanade.

La división del pipeline en:

```text
Carga
→ Preprocesamiento
→ Optical Flow
→ Métricas
```

permite medir cada etapa de forma independiente y facilita la identificación de los principales cuellos de botella.

El perfilado realizado sobre el baseline muestra que **Lucas–Kanade concentra la mayor parte del tiempo de cómputo**. Esto justifica la optimización SIMD mediante ARM NEON y posteriormente la descarga del mismo núcleo hacia otros aceleradores.

La separación clara entre baseline, optimización y aceleradores permite mantener una referencia común de correctitud y comparar de forma reproducible el rendimiento de las diferentes plataformas.
