# Introduction to Heterogeneous Computing

## Final Project - Topic: Optical Flow

Teacher: 

Luis G. León Vega

Students: 

  George Roussel Briceño Celestino
  
  Laura Elena Brenes Espinoza	           lauritabrenes@estudiantec.cr
  
  Marlon Méndez Naranjo              mnaranjo@estudiantec.cr

  Rodrigo Venegas Mora	           rovenegas@estudiantec.cr

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

## 3. Optimización Acelerada (ARM NEON)

### Planteamiento Técnico
El algoritmo de Lucas-Kanade calcula las derivadas parciales de imagen (Ix, Iy, It) y resuelve el sistema de ecuaciones 2x2 para cada ventana local alrededor de un píxel.

- **Versión Escalar (`lk_scalar.c`):** Opera un píxel a la vez mediante instrucciones **SISD**.
- **Versión Vectorial (`lk_neon.c`):** Utiliza registros de 128 bits (`float32x4_t`) mediante la librería `<arm_neon.h>` para procesar **4 píxeles en paralelo por ciclo de reloj**, acelerando la acumulación del Tensor de Estructura (ΣIx², ΣIy², ΣIxIy).
---

## 4. Resultados de Rendimiento en Kria KV260

Evaluando el par de frames reales de prueba (`1024x436` píxeles):

| Etapa del Pipeline | Tiempo Escalar (ms) | Tiempo NEON (ms) | Aceleración (Speedup) |
| :--- | :---: | :---: | :---: |
| **I/O Carga** | 109.37 | 109.37 | 1.0\times |
| **Preprocesamiento** | 92.35 | 92.35 | 1.0\times |
| **Núcleo Lucas-Kanade (LK)** | **365.93** | **147.57** | **2.48\times** |
| **Píxeles Válidos Resueltos** | 70.2\% | 70.2\% | Identidad funcional comprobada |

> **Conclusión del Benchmark:** La vectorización SIMD con ARM NEON reduce el tiempo de cómputo del algoritmo en **$\sim 60\%$**, logrando un **Speedup de $2.48\times$** sobre la CPU de la Kria KV260.
---
# Reporte de Optimizaciones - Prototipo en C

Este documento detalla las estrategias de optimización evaluadas y aplicadas en la implementación base en C del algoritmo de flujo óptico de Lucas-Kanade, cumpliendo con los requisitos de la evaluación del curso.

## 1. Optimizaciones Aplicadas

*   **Opciones del Compilador:** Se configuró CMake (`CMakeLists.txt`) para forzar el modo `Release` (equivalente a `-O3`), habilitando el desenrollado de bucles agresivo y la vectorización automática del compilador GCC. Adicionalmente, para arquitecturas ARM, se inyectó la bandera `-march=armv8-a+simd` para habilitar el conjunto de instrucciones NEON.
*   **Paralelismo a Nivel de Datos (SIMD):** Se aplicó vectorización explícita en `src/lk_neon.c`. Se utilizaron intrínsecas de ARM (`float32x4_t`, `vmlaq_f32`) para procesar 4 píxeles simultáneamente durante el cálculo de gradientes y la acumulación del Tensor de Estructura, logrando un speedup de ~2.48x sobre la Kria KV260.
*   **Ordenamiento de Datos (Separabilidad):** En la etapa de preprocesamiento (`src/preprocess.c`), el filtro Gaussiano 2D se implementó como dos pasadas ortogonales 1D (convolución separable). Esto reduce la complejidad de accesos a memoria de $O(K^2)$ a $O(2K)$ por píxel.
*   **Gestión Estática de Memoria:** Se eliminó la asignación dinámica (`malloc`/`free`) dentro de los bucles de cómputo. El orquestador (`main.c`) reserva todos los búferes (`OFImage`, `OFFlow`) una única vez y los reutiliza, reduciendo la fragmentación y los fallos de caché.

## 2. Optimizaciones Evaluadas y No Aplicadas (Con Justificación)

*   **Paralelismo a Nivel de Tareas (Multithreading CPU / OpenMP):** *No aplicado.* El algoritmo de Lucas-Kanade (al ser estrictamente local) es inherentemente paralelo por píxel. Distribuir este trabajo en los 4 hilos de la CPU (Cortex-A53 o Cortex-A57) mediante OpenMP introduciría sobrecarga de sincronización de contexto. Además, dado que el objetivo inmediato del proyecto es delegar el cómputo masivamente paralelo a la GPU (Jetson Nano, 128 CUDA cores) y a la FPGA (Kria KV260), optimizar la CPU a nivel de hilos resulta redundante y desvía recursos de las arquitecturas heterogéneas meta.
*   **Alineamiento Estricto de Datos (16-byte alignment):** *No aplicado rigurosamente.* Aunque se vectorizó con NEON, los punteros de memoria asignados por el sistema (vía `malloc` o `stb_image`) no están explícitamente forzados a un alineamiento de 16 bytes mediante atributos del compilador (`__attribute__((aligned(16)))`). Esto implica el uso de instrucciones de carga no alineada (`vld1q_f32` en lugar de cargas alineadas puras). No se aplicó porque el impacto en arquitecturas ARM modernas (ARMv8) es marginal, y requeriría reestructurar el asignador de memoria global.

## 3. Resumen de Siguientes Pasos (Estrategia de Mejora)

El perfilado estadístico (>100 muestras) demuestra que, tras aislar las penalizaciones de I/O (guardado CSV), la etapa 3 (`LK_COMPUTE`) es responsable de la práctica totalidad del tiempo de procesamiento.

**Estrategia a corto plazo (Acelerador I - Jetson Nano):**
Descargar la etapa 3 hacia la GPU integrada Maxwell. Se implementarán kernels en CUDA (`src/lk_cuda.cu`) para mapear el problema 2D directamente en un Grid de bloques de hilos, donde cada hilo resuelva la regla de Cramer para un píxel independiente. El objetivo es superar drásticamente los 45 ms actuales del baseline escalar.

## 5. Guía de Despliegue Rápido (Cheat Sheet)

### A. Sincronización desde la PC Local hacia la Kria
```bash
cd ~/Desktop/"Proyecto final"

rsync -avz --delete \
  --exclude 'build' --exclude '.git' --exclude 'results' \
  -e "ssh -J el5859@51.161.8.220:9222" \
  . el5859@192.168.0.82:~/"Proyecto final"
```

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
### C. Video demostrativo
https://github.com/user-attachments/assets/01c70c37-749b-45bd-bf76-07d737b1cba1

### Discord Channel
https://discord.com/channels/1342248894721167360/1536185160062468178
