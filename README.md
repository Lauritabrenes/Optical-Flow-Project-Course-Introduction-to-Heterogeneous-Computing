# Introduction to Heterogeneous Computing

## Final Project - Topic: Optical Flow

Teacher: 

Luis G. León Vega

Students: 

  George Roussel Briceño Celestino
  
  Laura Elena Brenes Espinoza	           lauritabrenes@estudiantec.cr
  
  Marlon Méndez Naranjo              mnaranjo@estudiantec.cr

  Rodrigo Venegas Mora	           rovenegas@estudiantec.cr

# Prototipo en C
Estructura del workspace
```text
Proyecto final/
├── CMakeLists.txt          # Configuración de compilación (flags -O3 y -march=armv8-a+simd)
├── include/
│   └── of.h                # Contrato global: estructuras OFImage, OFFlow y prototipos de función
├── src/
│   ├── main.c              # Punto de entrada, orquestación del pipeline y perfilado
│   ├── io.c                # Carga/guardado de imágenes (vía stb_image)
│   ├── preprocess.c        # Normalización de intensidades (uint8 -> float32)
│   ├── lk_scalar.c         # Algoritmo Lucas-Kanade secuencial en C (referencia)
│   ├── lk_neon.c           # Algoritmo Lucas-Kanade acelerado con intrínsecas SIMD ARM NEON
│   └── metrics.c           # Cálculo de EPE (End-Point Error) y estadísticas del flujo
├── data/frames/            # Par de fotogramas de prueba (.png)
├── scripts/
│   ├── profile.sh          # Automatización de N corridas del benchmark
│   └── analyze.py          # Cálculo de estadísticas (media, mediana, desviación)
└── results/profiling/      # Archivos CSV con los datos crudos del perfilado
```
---
## Arquitectura del Pipeline
El procesamiento en 4 etapas secuenciales. El orquestador (`main.c`) asigna la memoria una única vez para aislar el tiempo neto de cómputo en los kernels:
1. **I/O (`io.c`):** Carga de fotogramas RGB y conversión a escala de grises.
2. **Preprocesamiento (`preprocess.c`):** Normalización a `[0,1]` y filtrado Gaussiano separable.
3. **Cálculo (`lk_scalar.c` / `lk_neon.c`):** Extracción de gradientes espaciotemporales y cálculo del flujo.
4. **Métricas (`metrics.c`):** Evaluación del End-Point Error (EPE) y serialización de datos.
   
## 1. Optimización Acelerada (ARM NEON)

### Planteamiento Técnico
El algoritmo de Lucas-Kanade calcula las derivadas parciales de imagen (Ix, Iy, It) y resuelve el sistema de ecuaciones 2x2 para cada ventana local alrededor de un píxel.

- **Versión Escalar (`lk_scalar.c`):** Opera un píxel a la vez mediante instrucciones **SISD**.
- **Versión Vectorial (`lk_neon.c`):** Utiliza registros de 128 bits (`float32x4_t`) mediante la librería `<arm_neon.h>` para procesar **4 píxeles en paralelo por ciclo de reloj**, acelerando la acumulación del Tensor de Estructura (ΣIx², ΣIy², ΣIxIy).

## 2. Resultados de Rendimiento en Kria KV260

Evaluando el par de frames reales de prueba (`1024x436` píxeles):

| Etapa del Pipeline | Tiempo Escalar (ms) | Tiempo NEON (ms) | Aceleración (Speedup) |
| :--- | :---: | :---: | :---: |
| **I/O Carga** | 109.37 | 109.37 | 1.0\times |
| **Preprocesamiento** | 92.35 | 92.35 | 1.0\times |
| **Núcleo Lucas-Kanade (LK)** | **365.93** | **147.57** | **2.48\times** |
| **Píxeles Válidos Resueltos** | 70.2\% | 70.2\% | Identidad funcional comprobada |

> **Conclusión del Benchmark:** La vectorización SIMD con ARM NEON reduce el tiempo de cómputo del algoritmo en **$\sim 60\%$**, logrando un **Speedup de $2.48\times$** sobre la CPU de la Kria KV260.

## 3. Guía de Despliegue Rápido (Cheat Sheet)

### A. Sincronización desde la PC Local hacia la Kria
```bash
cd ~/Desktop/"Proyecto final"

rsync -avz --delete \
  --exclude 'build' --exclude '.git' --exclude 'results' \
  -e "ssh -J el5859@51.161.8.220:9222" \
  . el5859@192.168.0.82:~/"Proyecto final"
```
Dependencias necesarias en la tarjeta
sudo apt update && sudo apt install -y build-essential cmake python3

### B. Comilacion y ejecucion manual en la kria
Conectarse a la Kria
```
ssh -J el5859@51.161.8.220:9222 el5859@192.168.0.82
```
Compilar desde cero
```
cd ~/"Proyecto final"
mkdir -p results/profiling
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```
Ejecutar benchmark único
```
./bench_main ../data/frames/frame_000001.png ../data/frames/frame_000002.png
```
Ejecutar perfilado de 100 muestras (Ejemplo con NEON)
```
bash ../scripts/profile.sh ./bench_main ../data/frames/frame_000001.png ../data/frames/frame_000002.png neon 100
```
Analizar las métricas resultantes
```
python3 ../scripts/analyze.py ../results/profiling/times_neon.csv
```

### C. Video demostrativo
https://github.com/user-attachments/assets/01c70c37-749b-45bd-bf76-07d737b1cba1
---
# Prototipo Acelerador 1 (NVIDIA CUDA)

Esta etapa migra el núcleo de cómputo del algoritmo de Lucas-Kanade hacia la GPU utilizando NVIDIA CUDA. El objetivo es evaluar distintas estrategias de gestión de jerarquía de memoria y aislar el tiempo de ejecución de los kernels en la plataforma NVIDIA Jetson Nano (SoC Maxwell).

## Estructura del workspace (Archivos CUDA)

```text
Proyecto final/
├── include/
│   ├── cuda_check.h        # Macros para validación de errores en llamadas a CUDA API
│   ├── gpu_timing.h        # Definiciones para la medición fina con cudaEvent_t
│   └── of.h                # Contrato global actualizado con firmas de GPU
├── src/
│   ├── gpu_timing.cu       # Implementación de timers y warm-up del contexto CUDA
│   ├── lk_cuda.cu          # Baseline CUDA: 1 hilo = 1 píxel, lecturas directas a VRAM
│   ├── lk_cuda_opt.cu      # Kernel fusionado: cálculo al vuelo sin memoria compartida
│   ├── lk_cuda_shared.cu   # Acelerador final: carga cooperativa en caché L1 (__shared__)
│   └── main.c              # Orquestador actualizado con opciones --mode y warm-up
├── tests/
│   └── test_cuda_correctness.c # Validación automática de tolerancia de error (EPE) en GPU
├── scripts/
│   ├── profile_gpu.sh      # Perfilado con desglose interno (alloc/h2d/kernel/d2h/free)
│   └── profile.sh          # Automatización de corridas estadísticas generales
└── results/profiling/      # Salidas CSV de los perfilados en Jetson
```

## 1. Estrategias de Aceleración en GPU

Se implementaron tres enfoques para trasladar el cálculo a la tarjeta gráfica, atacando progresivamente los cuellos de botella de memoria:

*   **lk_cuda.cu**: básicamente es el baseline normal. Aquí cada hilo de la gpu toma un pixel d la imagen y hace el calculo. hace el recorrido de la ventana 7x7 de lucas kanade y calcula los gradientes, pero todo lo lee directamente de la memoria global (vram). Funciona bien y saca el resultado q es, pero es lento xq estar leyendo a cada rato de la global atraviesa todo el bus y genera un cuello de botella.
*   **lk_cuda_opt.cu**: aquí tratamos de optimizar fusionando todo el trabajo. En vez de guardar los gradientes, los calculamos en el kernel para ahorrar pasos. No se obtuvo lo que se pensó para sacar cada gradiente de la ventana cada hilo terminó haciendo 294 lecturas a la memoria global de nuevo. Parece que todos los hilos dependían de los datos al mismo tiempo, saturamos el anchode banda (memory bound full), entonces el tiempo mas bien empeoro en vez de mejorar.
*   **lk_cuda_shared.cu**: usando la memoria compartida de cuda (`__shared__`) los 256 hilos de un bloque cooperan entres si para cargar de una vez un pedazo d la imagen (un tile d 16x16 mas un halo d 4 pixeles para los bordes xq se ocupan para los gradientes, meten un cuadro d 24x24 total). Todo eso se sube a la cache L1 q es mas rápida, luego ponemos un `__syncthreads()` para q todos s esperen a que termine de cargarse el pedazo, y ahora los cálculos se hacen leyendo todo de la cache L1 en vez de ir hasta la ram de nuevo. Se atacó el problema del bus de raiz y bajo los tjempos, evitando leer la global 294 veces por pixel.

## 2. Resultados de Rendimiento en Jetson Nano

La validación funcional se realizó con `./test_cuda_correctness` sobre frames sintéticos, donde **todos los kernels CUDA pasaron la prueba de tolerancia** (max_err = 2.265e-06, 0% de píxeles fuera de tolerancia).

Evaluando los frames reales de prueba (`1024x436` píxeles) tras 100 corridas estadísticas, los tiempos para el **núcleo de cálculo puro** (`lk_compute_ms`) fueron:

| Arquitectura / Modo | Tiempo Promedio (ms) | Notas de hardware |
| :--- | :---: | :--- |
| **CPU (Scalar Baseline)** | 277.6 ms | Un hilo Cortex-A57 |
| **GPU (CUDA Baseline)** | 224.6 ms | Saturación del bus de memoria VRAM |
| **CPU (ARM NEON)** | 111.7 ms | Vectorización SIMD |
| **GPU (CUDA Shared)** | **91.98 ms** | Carga cooperativa en Caché L1 |

> **Análisis:** La memoria compartida (`cuda_shared`) logró vencer finalmente al núcleo ARM NEON, reduciendo el tiempo puro de cálculo a **~92 ms**. Sin embargo, aunque el kernel es superior, el tiempo total del pipeline (1343 ms) sigue estando dominado por la inicialización, la reserva de memoria (`cudaMalloc`) y las transferencias redundantes Host-to-Device y Device-to-Host (`cudaMemcpy`). Al ser la Jetson Nano un SoC que comparte físicamente la misma memoria LPDDR4, pagar el costo de estas transferencias es un desperdicio. Esto justifica que el siguiente paso de optimización requiera usar **Zero-Copy Memory** (memoria mapeada directamente) para eliminar el *overhead* del controlador.

## 3. Guía de Despliegue Rápido (Cheat Sheet)

### A. Dependencias en la Jetson Nano
```bash
sudo apt update && sudo apt install -y build-essential cmake python3 tree
```

### B. Sincronización (Rsync)
```bash
cd ~/Desktop/"Proyecto final"

rsync -avz --delete \
  --exclude 'build' --exclude '.git' --exclude 'results' \
  -e "ssh -J el5859@51.161.8.220:9222" \
  . el5859@192.168.0.182:~/"Proyecto final"
```

### C. Compilación y Validación Automática
Conectarse a la Jetson Nano y compilar desde cero:
```bash
ssh -J el5859@51.161.8.220:9222 el5859@192.168.0.182
cd ~/"Proyecto final"
mkdir -p results/profiling build && cd build
cmake ..
make -j$(nproc)

# Validar que ningún kernel tenga errores matemáticos
./test_cuda_correctness
```

### D. Perfilado y Análisis
Para medir el rendimiento estadístico aislando el tiempo total de cómputo del kernel con memoria compartida:
```bash
bash ../scripts/profile.sh ./bench_main ../data/frames/frame_000001.png ../data/frames/frame_000002.png cuda_shared 100
```

Si se desea ver el **desglose fino** de los tiempos internos de la GPU para auditar el cuello de botella (Alloc, H2D, Kernel, D2H, Free):
```bash
bash ../scripts/profile_gpu.sh ./bench_main ../data/frames/frame_000001.png ../data/frames/frame_000002.png cuda_shared 100
```

Analizar los resultados:
```bash
python3 ../scripts/analyze.py ../results/profiling/times_cuda_shared.csv
```

### Discord Channel
https://discord.com/channels/1342248894721167360/1536185160062468178
