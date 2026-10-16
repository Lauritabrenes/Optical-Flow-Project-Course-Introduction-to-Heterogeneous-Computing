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
