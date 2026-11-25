# Prototipo Acelerador II (AMD Kria KV260 – Vitis HLS)

Esta etapa traslada el núcleo de cómputo del algoritmo de Lucas–Kanade (LK) a la FPGA de la AMD Kria KV260 mediante un kernel en Vitis HLS (`lk_hls_v0`). El objetivo es usar un segundo acelerador, distinto de la GPU del Prototipo Acelerador I (Jetson Nano / CUDA), y perfilar en la placa el tiempo de cada etapa y el consumo de potencia.

`lk_hls_v0` es un port directo de `lk_scalar.c`:

- **Gradientes:** calcula `Ix`, `Iy` e `It` con diferencias centrales y los guarda en URAM.
- **Ventana 7×7:** acumula Σ Ix², Iy², IxIy, IxIt e IyIt.
- **Sistema 2×2:** lo resuelve con la regla de Cramer y escribe `u` y `v` en DDR.
- **Interfaz:** 4 puertos `m_axi` (f1, f2, u, v) conectados a HP0–HP3 y un puerto de control `s_axilite` (w, h, start/done) en `0xA0000000`.
- **Límite:** 320×240, porque URAM ocupa 63 de 64 bloques. Por eso no llega a 720p.

---

## Estructura del entregable

```
Prototipo_acelerador_II/
├── 0_src/
│   ├── README.md
│   ├── hls/include/lk_hls.h              # Parámetros (ventana 7x7, tamaño máximo)
│   ├── hls/v0_port_directo/lk_hls_v0.cpp # Kernel v0
│   ├── scripts/run_hls_v0.tcl            # Flujo HLS por script
│   └── tb/tb_lk_hls.cpp                  # Testbench contra referencia (corregido para co-sim)
├── 1_hls_config/hls_config.cfg           # part xck26, reloj 5 ns, -DLK_MAX_W=320 -DLK_MAX_H=240
├── 2_csim/lk_hls_v0_csim.log             # Simulación en C (PASS)
├── 3_csynth/csynth.rpt                   # Reporte de síntesis HLS
├── 4_cosim/                              # Co-simulación RTL (Verilog, PASS)
├── 5_vivado_impl/                        # Potencia, timing, utilización, ruteo, runme.log
├── 6_bitstream/                          # design_1_wrapper.bit + design_1.hwh
└── 7_board/                              # Ejecución y perfilado en la Kria
    ├── lk_v0.bit / lk_v0.hwh             # Overlay PYNQ (deben llamarse igual)
    ├── frames/                           # frame_000001/2_320x240_crop.png
    ├── run_lk.py, run.sh                 # Primera prueba funcional (10 corridas)
    ├── run_lk_prof.py                    # Perfilado: 3 de calentamiento + N corridas
    ├── run_prof.sh                       # Versión con platformstats (NO usar en Kria 83)
    ├── sample_power.sh                   # Muestreo de potencia vía hwmon (cada 0.2 s)
    ├── power_summary.py                  # Potencia en reposo/carga y energía por frame
    ├── list_ips.py                       # Lista las IP del overlay
    ├── times_fpga_v0.csv                 # Una fila por corrida (300)
    ├── stats_fpga_v0.csv                 # Media, mediana, std, min, max por etapa
    ├── summary_fpga_v0.csv               # Resumen: kernel, FPS, ciclos/píxel, error
    ├── platformstats_idle.txt / _run.txt # Muestras de potencia
    ├── prof_fpga_v0.log                  # Log en pantalla del perfilado
    └── flow_000001.npy                   # Flujo (u, v) resultante
```

---

## 1. Decisiones de aceleración y descarga de trabajo

- El perfilado del prototipo en C mostró que el cálculo LK (gradientes, ventana y solución 2×2) domina el tiempo del pipeline. Por eso se descarga a la FPGA, igual que se hizo en CUDA.
- En el Acelerador I, el tiempo total en la Jetson quedó dominado por `cudaMalloc` y `cudaMemcpy`. En la Kria, los buffers se reservan una sola vez con `pynq.allocate` (memoria CMA contigua compartida con el PS). Así, `h2d` y `d2h` se reducen a sincronizar la caché (unos 0.2 ms y 0.14 ms).
- v0 es un port directo sin optimizaciones de pipeline o *line buffer*. Sirve como baseline de la FPGA para la comparación heterogénea.

---

## 2. Resultados

### 2.1 Validación y síntesis

| Prueba | Resultado |
|---|---|
| C-sim (frames sintéticos 128×96, dx=1.0, dy=0.5) | PASS, `max_err = 0`, 0 % de píxeles fuera de tolerancia |
| Co-sim RTL (Verilog) | PASS, 172 945 ciclos |
| Ruteo | 133 248 redes ruteadas, 0 errores |
| Utilización | LUT 72.80 %, FF 30.81 %, DSP 379 (30.37 %), URAM 63/64 (98.44 %), BRAM 1 |
| Reloj | Objetivo HLS 5 ns; PL a 142.86 MHz en la placa |

### 2.2 Perfilado en la placa (Kria KV260, 320×240, 300 corridas + 3 de calentamiento)

| Etapa | Media (ms) | Mediana (ms) | Std (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|---|
| `load_ms` (PNG + escala) | 24.002 | 23.918 | 0.272 | 23.652 | 24.952 |
| `h2d_ms` | 0.215 | 0.210 | 0.023 | 0.202 | 0.388 |
| `kernel_ms` | **22.530** | 22.500 | 0.088 | 22.485 | 22.915 |
| `d2h_ms` | 0.149 | 0.128 | 0.046 | 0.120 | 0.293 |
| `total_ms` | 46.895 | 46.779 | 0.306 | 46.481 | 47.982 |

| Métrica | Valor |
|---|---|
| FPS del kernel | 44.39 |
| Ciclos por píxel | 41.91 |
| Error máximo u / v vs. referencia NumPy (ventana 7×7) | 3.11 / 4.39 |
| Error absoluto medio | 0.0016 |
| Potencia en reposo (45 muestras) | 3887 mW |
| Potencia en carga (101 muestras) | 4166 mW (máx. 4580 mW) |
| Potencia dinámica | ≈ 279 mW |
| Energía dinámica por frame | ≈ 6.3 mJ |

**Análisis:** el kernel representa cerca del 48 % del tiempo total y las transferencias menos del 1 %. El resto es carga de la imagen en el CPU (`load_ms`). Por tanto, el siguiente cuello de botella es el propio kernel v0 (41.9 ciclos por píxel) y la lectura de los frames en el host, no la transferencia de datos.

> Nota: `prof_fpga_v0.log` corresponde a una corrida anterior (kernel 22.510 ms). Las cifras oficiales son las de los CSV.

---

## 3. Guía de despliegue rápido

> Comandos marcados con `(PC)` se ejecutan en el equipo local (Manjaro) y con `(Kria)` dentro de la placa. **No incluir la contraseña del laboratorio en el repositorio, scripts ni README.**

### A. Construcción del hardware (Vitis HLS + Vivado)
1. Abrir `1_hls_config/hls_config.cfg` en Vitis HLS (o ejecutar `0_src/scripts/run_hls_v0.tcl`). Ajustar las rutas de `syn.file` y `tb.file` a `0_src/` si se usa esta carpeta de forma independiente.
2. Ejecutar C-sim → C-synth → Co-sim → Export IP.
3. En Vivado: Zynq UltraScale+ MPSoC + `lk_hls_v0` (m_axi en HP0–HP3, s_axilite en `0xA0000000`), generar bitstream.
4. Copiar `design_1_wrapper.bit` y `design_1.hwh` como `lk_v0.bit` y `lk_v0.hwh` (mismo nombre base).

### B. Subir los archivos a la Kria (PC)
```bash
B=/Prototipo_acelerador_II/7_board
rsync -avz -e "ssh -J ecaslab" $B/lk_v0.bit $B/lk_v0.hwh $B/frames \
      $B/run_lk_prof.py $B/sample_power.sh $B/power_summary.py \
      el5859@192.168.0.83:~/deploy/
md5sum $B/lk_v0.bit
```

### C. Verificación en la placa (Kria)
```bash
ssh kria83
cd ~/deploy && sed -i 's/\r$//' run_lk_prof.py sample_power.sh power_summary.py
md5sum lk_v0.bit                                       # igual al del PC
grep -o 'BASEVALUE="0xA0000000"[^>]*lk_hls_v0_0' lk_v0.hwh | head -1
chmod +x sample_power.sh
sudo ./sample_power.sh test.txt 2 && cat test.txt     # ~10 líneas, ~3900 mW
```

### D. Prueba rápida (10 corridas, Kria)
```bash
sudo bash -c 'source /etc/profile.d/pynq_venv.sh; source /etc/profile.d/xrt_setup.sh 2>/dev/null; cd /home/el5859/deploy && python3 run_lk_prof.py 10'
```
Esperado: `kernel_ms ≈ 22.5`, `mean_abs_err ≈ 0.0016`.

### E. Perfilado completo con potencia (Kria, un comando a la vez)
```bash
sudo ./sample_power.sh platformstats_idle.txt 10       # placa en reposo 10 s

sudo ./sample_power.sh platformstats_run.txt 0 & SP=$!; \
sudo bash -c 'source /etc/profile.d/pynq_venv.sh; source /etc/profile.d/xrt_setup.sh 2>/dev/null; cd /home/el5859/deploy && python3 run_lk_prof.py 300'; \
sudo kill $SP

python3 power_summary.py
```
> No usar `run_prof.sh` en la Kria 83: llama a `platformstats`, que no está instalado. Usar `sample_power.sh`.

### F. Bajar resultados (PC)
```bash
rsync -avz -e "ssh -J ecaslab" \
  'el5859@192.168.0.83:~/deploy/*_fpga_v0.csv' \
  'el5859@192.168.0.83:~/deploy/platformstats_*.txt' \
  'el5859@192.168.0.83:~/deploy/flow_000001.npy' "$B"/
```

### Valores esperados

| Métrica | Valor |
|---|---|
| `kernel_ms` | 22.5 ± 0.1 |
| `fps_kernel` | ~44.4 |
| `cycles_per_pixel` | ~41.9 |
| `mean_abs_err` | ~0.0016 |
| NaN en el flujo | 0 |
| Potencia reposo / carga | ~3890 / ~4170 mW |
| Potencia dinámica | ~280–300 mW |

### Problemas comunes

| Síntoma | Causa / solución |
|---|---|
| TIMEOUT o placa congelada | `.bit` y `.hwh` no coinciden: comparar `md5sum` y repetir el `grep` |
| `Could not find IP lk_hls_v0_0` | Usar `MMIO(0xA0000000, 0x10000)` en lugar de `getattr` |
| `t.xclbin not found` | Se usó `sudo -E`; usar `sudo bash -c '...'` |
| `$'\r': command not found` | Repetir el `sed` del paso C |
| Potencia sin muestras | No corrió `sample_power.sh` o los nombres de archivo no coinciden |
| `Could not resolve hostname ecaslab` | El `rsync` se corrió dentro de la Kria; correrlo desde el PC |

---
### Video demostrativo(Prototipo Acelerador II)
https://github.com/user-attachments/assets/29abb0d5-2d04-459b-95e4-89a06c8dac4c

## 4. Siguientes pasos

- **Optimizar el kernel:** `#pragma HLS PIPELINE II=1` en los bucles de píxel, *line buffers* y ventana deslizante en lugar de guardar los gradientes completos en URAM. Esto libera URAM y permite resoluciones mayores.
- **Ancho de bus:** empaquetar píxeles (`ap_uint<128>`) en `m_axi` y usar ráfagas.
- **Tipos de datos:** evaluar punto fijo (`ap_fixed`) en la acumulación para reducir el uso de DSP y LUT.
- **Host:** reducir `load_ms` decodificando los frames una sola vez o en un hilo paralelo al kernel.
- **Análisis heterogéneo:** comparar CPU (NEON), GPU (CUDA shared) y FPGA a la misma resolución, en tiempo, energía por frame y utilización.
