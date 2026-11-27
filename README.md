# Introduction to Heterogeneous Computing

## Final Project - Topic: Optical Flow

Teacher: 

Luis G. León Vega

Students: 

  George Roussel Briceño Celestino        roussel@estudiantec.cr
  
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

**Análisis:** La memoria compartida (`cuda_shared`) logró vencer finalmente al núcleo ARM NEON, reduciendo el tiempo puro de cálculo a **~92 ms**. Sin embargo, aunque el kernel es superior, el tiempo total del pipeline (1343 ms) sigue estando dominado por la inicialización, la reserva de memoria (`cudaMalloc`) y las transferencias redundantes Host-to-Device y Device-to-Host (`cudaMemcpy`). Al ser la Jetson Nano un SoC que comparte físicamente la misma memoria LPDDR4, pagar el costo de estas transferencias es un desperdicio. Esto justifica que el siguiente paso de optimización requiera usar **Zero-Copy Memory** (memoria mapeada directamente) para eliminar el *overhead* del controlador.

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
chmod +x ../scripts/profile.sh
```
o chmod +x ../scripts/profile_gpu.sh 

### D. Perfilado y Análisis
Para medir el rendimiento estadístico aislando el tiempo total de cómputo del kernel baseline(sin optimizaciones):
```bash
../scripts/profile.sh ./bench_main ../data/frames/frame_000001.png ../data/frames/frame_000002.png cuda 100
```
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
### E. Video demostrativo
https://github.com/user-attachments/assets/212f186c-e63a-4ccf-9f28-54bf75299faa

---
# Prototipo Acelerador 2 (AMD Kria KV260 - Vitis HLS)

Esta etapa traslada el núcleo de cómputo de Lucas-Kanade a la FPGA de la AMD Kria KV260 mediante un kernel en Vitis HLS (`lk_hls_v0`). El objetivo es usar un segundo acelerador, distinto de la GPU del Acelerador 1, y perfilar en la placa el tiempo de cada etapa y el consumo de potencia.

`lk_hls_v0` es un port directo de `lk_scalar.c`: mismo algoritmo (gradientes centrales, ventana 7x7, regla de Cramer) para poder validarlo contra la CPU. No tiene optimizaciones de pipeline ni *line buffers*, sirve como baseline de la FPGA.

**Resolución de trabajo:** 320x240. Es un **recorte de los mismos frames** `frame_000001.png` y `frame_000002.png` (Sintel, 1024x436) que se usaron en la CPU y en la Jetson, así que el contenido de la imagen es el mismo y solo cambia el tamaño. No se pudo ir más arriba porque v0 guarda Ix, Iy e It completos en URAM y a 320x240 ya ocupa 63 de los 64 bloques.

## Estructura del workspace (Archivos FPGA)

```text
Prototipo_acelerador_II/
├── 0_src/
│   ├── hls/include/lk_hls.h              # Parámetros (ventana 7x7, tamaño máximo)
│   ├── hls/v0_port_directo/lk_hls_v0.cpp # Kernel v0
│   ├── scripts/run_hls_v0.tcl            # Flujo HLS por script (legado, ver nota en la guía)
│   └── tb/tb_lk_hls.cpp                  # Testbench contra referencia (corregido para co-sim)
├── 1_hls_config/hls_config.cfg           # part xck26, reloj 5 ns, -DLK_MAX_W=320 -DLK_MAX_H=240
├── 2_csim/lk_hls_v0_csim.log             # Simulación en C (PASS)
├── 3_csynth/csynth.rpt                   # Reporte de síntesis HLS
├── 4_cosim/                              # Co-simulación RTL (Verilog, PASS)
├── 5_vivado_impl/                        # Potencia, timing, utilización, ruteo, runme.log
├── 6_bitstream/                          # design_1_wrapper.bit + design_1.hwh
└── 7_board/                              # Ejecución y perfilado en la Kria
    ├── lk_v0.bit / lk_v0.hwh             # Overlay PYNQ (mismo nombre base, misma compilación)
    ├── frames/                           # frame_000001/2_320x240_crop.png
    ├── run_lk.py, run.sh                 # Primera prueba funcional
    ├── run_lk_prof.py                    # Perfilado: 3 de calentamiento + N corridas
    ├── sample_power.sh                   # Muestreo de potencia vía hwmon (INA260, cada 0.2 s)
    ├── power_summary.py                  # Potencia en reposo/carga y energía por frame
    ├── list_ips.py                       # Lista las IP del overlay
    └── results/                          # CSV de tiempos, potencia, log y flujo (u, v)
```

## 1. Decisiones de Aceleración y Descarga de Trabajo

*   **Qué se descarga:** igual que en CUDA, el perfilado del prototipo en C mostró que el cálculo LK (gradientes, ventana y solución 2x2) domina el tiempo del pipeline, entonces es lo que se pasa a la FPGA. La carga de la imagen y la escala se quedan en el ARM.
*   **Interfaz del kernel:** 4 puertos `m_axi` (f1, f2, u, v) conectados a HP0-HP3 para leer y escribir directo en la DDR, y un puerto de control `s_axilite` (w, h, start/done) en `0xA0000000`.
*   **Memoria:** en la Jetson el tiempo total se lo comían `cudaMalloc` y `cudaMemcpy`. En la Kria los buffers se reservan una sola vez con `pynq.allocate` (memoria CMA contigua compartida con el PS), así que `h2d` y `d2h` quedan en sincronizar la caché (~0.2 ms y ~0.15 ms).
*   **Gradientes en URAM:** `Ix`, `Iy` e `It` se guardan completos en URAM. Es lo más directo de portar, pero es lo que limita la resolución a 320x240.

## 2. Resultados de Rendimiento en Kria KV260 (FPGA)

### Validación y síntesis

| Prueba | Resultado |
| :--- | :--- |
| C-sim (frames sintéticos 128x96, dx=1.0, dy=0.5) | PASS, `max_err = 0`, 0 % de píxeles fuera de tolerancia |
| Co-sim RTL (Verilog) | PASS, 172 945 ciclos |
| Ruteo | 133 248 redes ruteadas, 0 errores |
| Utilización | LUT 72.80 %, FF 30.81 %, DSP 379 (30.37 %), URAM 63/64 (98.44 %), BRAM 1 |
| Reloj | Objetivo HLS 5 ns; PL a 142.86 MHz en la placa |

### Perfilado en la placa (320x240, 300 corridas + 3 de calentamiento)

| Etapa | Media (ms) | Mediana (ms) | Std (ms) | Min (ms) | Max (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `load_ms` (PNG + escala) | 24.002 | 23.918 | 0.272 | 23.652 | 24.952 |
| `h2d_ms` | 0.215 | 0.210 | 0.023 | 0.202 | 0.388 |
| **`kernel_ms`** | **22.530** | 22.500 | 0.088 | 22.485 | 22.915 |
| `d2h_ms` | 0.149 | 0.128 | 0.046 | 0.120 | 0.293 |
| `total_ms` | 46.895 | 46.779 | 0.306 | 46.481 | 47.982 |

| Métrica | Valor |
| :--- | :---: |
| FPS del kernel | 44.39 |
| Ciclos por píxel | 41.91 |
| Error absoluto medio vs. referencia NumPy (ventana 7x7) | 0.0016 px |
| Error máximo u / v (píxeles aislados) | 3.11 / 4.39 px |
| Potencia en reposo / en carga | 3887 mW / 4166 mW (máx. 4580 mW) |
| Potencia dinámica | ~279 mW |
| Energía dinámica por frame | ~6.3 mJ |

**Análisis:** el kernel es cerca del 48 % del tiempo total y las transferencias no llegan ni al 1 %, la diferencia con la Jetson es grande porque aquí no hay copias, solo sincronización de caché. El otro ~51 % es la carga del PNG en el ARM (`load_ms`). Los ~42 ciclos por píxel salen casi igual a las 49 iteraciones de la ventana 7x7, o sea que v0 está limitado por cómputo serial y no por memoria: cada píxel recorre su ventana completa uno por uno. Por eso el siguiente cuello de botella es el propio kernel y la lectura de los frames en el host, no la transferencia de datos.

## 3. Guía de Despliegue Rápido (Cheat Sheet)

> En la Kria se usa la cuenta compartida `el5859` (Kria 83, `192.168.0.83`). No poner la contraseña del laboratorio en el repo, scripts ni README.

### A. Construcción del hardware (Vitis HLS + Vivado, PC)
1. Ajustar las rutas `syn.file` y `tb.file` de `1_hls_config/hls_config.cfg` a la carpeta `0_src/`.
2. Correr C-sim, C-synth, Co-sim y exportar el IP (`ip_catalog`). Con Vitis unificado (2024.2 o más nuevo):
```bash
v++ -c --mode hls --config 1_hls_config/hls_config.cfg --work_dir proj_lk_v0
```
   El `run_hls_v0.tcl` es del flujo viejo (`vitis_hls`): exporta `.xo` y no pasa `-DLK_MAX_W=320 -DLK_MAX_H=240`, así que si se usa hay que corregir eso.
3. En Vivado: Zynq UltraScale+ MPSoC + `lk_hls_v0` (m_axi en HP0-HP3, s_axilite en `0xA0000000`), validar y generar el bitstream.
4. Copiar `design_1_wrapper.bit` y `design_1.hwh` como `lk_v0.bit` y `lk_v0.hwh` (mismo nombre base, misma compilación).

### B. Sincronización hacia la Kria (PC)
```bash
cd ~/Desktop/"Proyecto final"/Prototipo_acelerador_II/7_board

rsync -avz -e "ssh -J el5859@51.161.8.220:9222" \
  lk_v0.bit lk_v0.hwh frames run_lk_prof.py sample_power.sh power_summary.py \
  el5859@192.168.0.83:~/deploy/
md5sum lk_v0.bit
```
La cuenta es compartida, esto sobrescribe `~/deploy`. Si otro compañero tiene resultados ahí, avisar antes. Desde Windows (sin `rsync`) se puede usar `scp -J el5859@51.161.8.220:9222 ...` con los mismos archivos.

### C. Preparación y verificación en la Kria
```bash
ssh -J el5859@51.161.8.220:9222 el5859@192.168.0.83
cd ~/deploy && sed -i 's/\r$//' run_lk_prof.py sample_power.sh power_summary.py
chmod +x sample_power.sh
md5sum lk_v0.bit                                       # igual al del PC
grep -o 'BASEVALUE="0xA0000000"[^>]*lk_hls_v0_0' lk_v0.hwh | head -1
for h in /sys/class/hwmon/hwmon*; do echo "$h $(cat $h/name)"; done   # debe salir hwmon2 ina260_u14
sudo ./sample_power.sh test.txt 2 && cat test.txt     # ~10 líneas, ~3900 mW
```
Si el sensor sale en otro `hwmon` (por ejemplo `hwmon3`), ajustar el script con `sed -i 's/hwmon2/hwmon3/' sample_power.sh`.

### D. Prueba rápida (10 corridas, Kria)
```bash
sudo bash -c 'source /etc/profile.d/pynq_venv.sh; source /etc/profile.d/xrt_setup.sh 2>/dev/null; cd /home/el5859/deploy && python3 run_lk_prof.py 10'
```
Esperado: `kernel_ms ≈ 22.5`, `mean_abs_err ≈ 0.0016`. Si pasan ~10 s sin imprimir nada, `Ctrl+C` y revisar la tabla de problemas.

### E. Perfilado completo con potencia (Kria, un comando a la vez)
```bash
sudo ./sample_power.sh platformstats_idle.txt 10       # placa en reposo 10 s, no usarla mientras

sudo ./sample_power.sh platformstats_run.txt 0 & SP=$!; \
sudo bash -c 'source /etc/profile.d/pynq_venv.sh; source /etc/profile.d/xrt_setup.sh 2>/dev/null; cd /home/el5859/deploy && python3 run_lk_prof.py 300'; \
sudo kill $SP

python3 power_summary.py
```
No usar `run_prof.sh` en la Kria 83: llama a `platformstats`, que no está instalado. Se usa `sample_power.sh`.

### F. Bajar resultados (PC)
```bash
rsync -avz -e "ssh -J el5859@51.161.8.220:9222" \
  'el5859@192.168.0.83:~/deploy/*_fpga_v0.csv' \
  'el5859@192.168.0.83:~/deploy/platformstats_*.txt' \
  'el5859@192.168.0.83:~/deploy/flow_000001.npy' results/
```

### Valores esperados

| Métrica | Valor |
| :--- | :---: |
| `kernel_ms` | 22.5 ± 0.1 |
| `fps_kernel` | ~44.4 |
| `cycles_per_pixel` | ~41.9 (a 142.86 MHz) |
| `mean_abs_err` | ~0.0016 px |
| NaN en el flujo | 0 |
| Potencia reposo / carga | ~3890 / ~4170 mW |
| Potencia dinámica | ~280-300 mW |

### Problemas comunes

| Síntoma | Causa / solución |
| :--- | :--- |
| TIMEOUT o placa congelada | `.bit` y `.hwh` no coinciden: comparar `md5sum` y repetir el `grep` |
| `Could not find IP lk_hls_v0_0` | Usar `MMIO(0xA0000000, 0x10000)` en lugar de `getattr` |
| `t.xclbin not found` | Se usó `sudo -E`; usar `sudo bash -c '...'` |
| `$'\r': command not found` | Repetir el `sed` del paso C |
| Potencia sin muestras | No corrió `sample_power.sh` o los nombres de archivo no coinciden |
| `Could not resolve hostname` | El `rsync` se corrió dentro de la Kria; correrlo desde el PC |

### G. Video demostrativo(Prototipo Acelerador II)
https://github.com/user-attachments/assets/29abb0d5-2d04-459b-95e4-89a06c8dac4c

## 4. Siguientes Pasos

*   **Optimizar el kernel:** cambiar los gradientes completos en URAM por *line buffers* de 7 filas y sumas de ventana incrementales. Con eso baja de ~42 ciclos por píxel hacia 1-2, se libera la URAM y se puede correr a 1024x436 igual que la CPU y la GPU. Solo poner `#pragma HLS PIPELINE` no alcanza, porque Vitis ya pipelinea el lazo interno de la ventana.
*   **Ancho de bus:** empaquetar píxeles (`ap_uint<128>`) en `m_axi` y usar ráfagas.
*   **Tipos de datos:** probar punto fijo (`ap_fixed`) en la acumulación para bajar DSP y LUT.
*   **Host:** bajar `load_ms` decodificando los frames una sola vez o en un hilo en paralelo al kernel.
*   **Análisis heterogéneo:** ya se hizo la comparación de CPU (NEON), GPU (CUDA shared) y FPGA con el mismo recorte de 320x240 (ver la sección de abajo). Falta sumarle energía por frame de la Jetson cuando se habilite su sensor de potencia.

# Comparación entre plataformas (mismo recorte 320x240)

Para comparar las tres plataformas de forma justa, se corrieron todas las versiones con el **mismo recorte 320x240 de los mismos frames** (`frame_000001/2_320x240_crop.png`), 100 corridas por modo. Se compara el **núcleo puro de Lucas-Kanade** (`lk_compute_ms` en CPU/GPU, `kernel_ms` en la FPGA) y no el tiempo total, porque el pipeline en C agrega un blur gaussiano y un guardado de CSV que el host de la FPGA no ejecuta.

| Placa / modo        | Núcleo LK (ms) | ns/píxel | Total (ms) |
| :--- | :---: | :---: | :---: |
| Jetson ARM NEON     | **12.42** | **162** | 238.1 |
| Jetson CUDA shared  | 20.03 | 261 | 233.4 |
| Kria ARM NEON       | 21.18 | 276 | 278.8 |
| FPGA `lk_hls_v0`    | 22.53 | 293 | 46.9 |
| Jetson scalar       | 26.42 | 344 | 248.5 |
| Kria scalar         | 32.35 | 421 | 279.4 |
| Jetson CUDA base    | 128.81 | 1677 | 338.3 |

**Lectura:**

*   La FPGA v0 es **1.44x más rápida que la CPU escalar** de la misma Kria, pero la **NEON de la misma Kria le gana por poco** (21.2 vs 22.5 ms). Esto confirma que v0 está limitada por cómputo serial (~42 ciclos por píxel ≈ 49 iteraciones de la ventana) y deja claro el margen para la versión con *line buffers*.
*   A este tamaño chico, **la NEON de la Jetson es lo más rápido de todo**. El kernel CUDA shared, que ganaba a 1024x436, aquí pierde contra NEON porque la imagen es pequeña y el costo fijo de lanzar el kernel y copiar (H2D ~3.3 ms, D2H ~6.5 ms) pesa más que el cómputo. El CUDA base además incluye crear el contexto CUDA en cada corrida (sin *warm-up*).
*   En la FPGA las transferencias son **menos del 1 %** del tiempo porque el host y la FPGA comparten la DDR; en la Jetson sí se paga H2D/D2H. Por eso el `total` de la FPGA (46.9 ms) es mucho menor que el de CPU/GPU, aunque su núcleo no sea el más rápido.
*   Todos los modos dan el mismo flujo (84.9 % de píxeles con solución), así que la comparación es sobre el mismo resultado numérico.

**Potencia (solo Kria, sensor INA260):** reposo 3886 mW, scalar 4024 mW, NEON 4003 mW, FPGA v0 4166 mW. La Jetson `.182` no expone su sensor de potencia, así que de esa placa no hay energía por frame medida. La potencia de CPU es el promedio de toda la corrida (el núcleo LK es una fracción del tiempo, domina el guardado de CSV), así que subestima la potencia durante el cálculo.

# Prueba de video (demo multiplataforma, 320x240)

Para confirmar que las tasas se sostienen sobre una secuencia real y no solo sobre un par de cuadros, se corrió el **mismo kernel** sobre dos clips cortos a **320x240**, en las 6 plataformas:

*   **Pong** — 150 cuadros, clip sintético (movimiento lento con textura suave).
*   **alley_1** — 50 cuadros de la escena `alley_1` de MPI-Sintel.

Todas las plataformas dan **el mismo flujo** (0 % de píxeles con diferencia > 0.05 respecto a la referencia escalar), y el orden de rendimiento se mantiene igual que con el par de cuadros.

| Plataforma | Pong LK (ms) | Pong FPS pipeline | alley_1 LK (ms) | alley_1 FPS pipeline |
| :--- | :---: | :---: | :---: | :---: |
| Jetson ARM NEON | 9.25 | 54.5 | 10.04 | 43.5 |
| Kria ARM NEON | 18.75 | 33.0 | 19.36 | 28.8 |
| Jetson CUDA shared | 19.66 (kernel 13.04) | 34.8 | 20.41 (kernel 13.04) | 30.0 |
| FPGA `lk_hls_v0` | 19.92 (kernel 19.38) | 24.2 | 22.45 (kernel 21.90) | 21.6 |
| Jetson scalar | 24.30 | 30.0 | 24.58 | 25.0 |
| Kria scalar | 29.25 | 21.8 | 30.46 | 21.8 |

Notas:

*   `LK (ms)` = tiempo medio del núcleo Lucas-Kanade por cuadro; `kernel` = solo el kernel (en CUDA shared y FPGA, sin las transferencias).
*   Los tiempos salen un poco menores que en la tabla del par de cuadros porque el proceso queda "caliente" a lo largo de la secuencia.
*   La potencia del video **no** se usa para energía (mezcla la recarga del bitstream y la verificación en NumPy); para energía se usa la medición del perfilado en placa de arriba.
*   Rejilla comparativa de las 6 plataformas en `docs/videos/demo_pong_grid.mp4` y `docs/videos/demo_alley_1_grid.mp4`. Este demo es para la defensa.

# Declaración de uso de IA

Durante el desarrollo del proyecto se usaron herramientas de inteligencia artificial como apoyo en la programación, la depuración, la generación de scripts de perfilado y la redacción y traducción del artículo y de este README. Las decisiones de diseño, el particionamiento de hardware, la integración y la validación de todos los resultados fueron realizadas por los autores.

Las evidencias de estas consultas están en el repositorio:

*   Etapa FPGA (Acelerador II): `Prototipo_acelerador_II/AWS Quick AI usage.pdf`, registro de uso de IA de Marlon para esta etapa.
*   `docs/evidencias/Rodrigo interaction log.md` y `docs/evidencias/Marlon_ai_interaction_log.md`, registros del equipo de etapas previas.
*   Enlaces Microsoft Copilot:
  https://copilot.cloud.microsoft/chat/share/eyJzaGFyZUlkIjoiNDQyZGY4YzItOWQ0Yi00NzliLTk1ZTQtYzkwZTUzYTVjYmZiIiwiY29udmVyc2F0aW9uSWQiOiI3YTk4MDk3NC03ZjMyLTRhMTctYmRiZC0wODliM2M5Y2Q3Y2IifQ%3D%3D
*   El artículo incluye además su propio enlace de consulta en la sección *AI Support in Development and Declaration*.

# Artículo científico en construcción
Lo puede encontrar en Optical-Flow-Project-Course-Introduction-to-Heterogeneous-Computing
/Articulo Cientifico/main.pdf

### Discord Channel
https://discord.com/channels/1342248894721167360/1536185160062468178
