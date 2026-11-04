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

Esta etapa migra el núcleo de cómputo del algoritmo de Lucas-Kanade hacia la GPU utilizando CUDA. El objetivo es evaluar distintas estrategias de gestión de jerarquía de memoria sobre la plataforma NVIDIA Jetson Nano (SoC Maxwell).

## Estructura del workspace (Archivos CUDA)

```text
Proyecto final/
├── CMakeLists.txt          # Configuración (cmake_minimum_required 3.10 y target nvcc)
├── include/
│   └── of.h                # Contrato global actualizado con firmas de funciones CUDA
├── src/
│   ├── main.c              # Orquestador actualizado con argumentos CLI (--mode cuda_shared)
│   ├── lk_cuda.cu          # Baseline CUDA: 1 hilo = 1 píxel, lecturas directas a VRAM (DRAM)
│   ├── lk_cuda_opt.cu      # Kernel fusionado: Cálculo al vuelo sin memoria compartida
│   └── lk_cuda_shared.cu   # Acelerador final: Carga cooperativa en Memoria Compartida (Caché L1)
├── data/frames/            # Par de fotogramas de prueba (.png)
├── scripts/
│   ├── profile.sh          # Automatización de N corridas aislando tiempos por modo
│   └── analyze.py          # Análisis estadístico del rendimiento
└── results/profiling/      # Archivos CSV con los datos crudos del perfilado en la Jetson


### Discord Channel
https://discord.com/channels/1342248894721167360/1536185160062468178
