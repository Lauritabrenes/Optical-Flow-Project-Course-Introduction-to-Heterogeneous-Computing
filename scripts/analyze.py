#!/usr/bin/env python3
# ============================================================================
#  scripts/analyze.py
# ----------------------------------------------------------------------------
#  Analisis estadistico de los tiempos de perfilado generados por profile.sh.
#
#  Lee un CSV con la cabecera:
#      run,io_ms,preprocess_ms,lk_scalar_ms,save_ms,total_ms
#  y calcula, para cada etapa del pipeline:
#      - Media, Mediana, Desviacion estandar, Minimo, Maximo
#
#  Uso:
#      ./analyze.py [PATH]
# ============================================================================

import csv
import os
import statistics
import sys

# ----------------------------------------------------------------------------
#  0. Ruta de entrada
# ----------------------------------------------------------------------------
path = sys.argv[1] if len(sys.argv) > 1 else "../results/profiling/times.csv"

# ----------------------------------------------------------------------------
#  1. Validacion de la ruta de entrada    [MEJORA 1]
# ----------------------------------------------------------------------------
if not os.path.exists(path):
    print(f"ERROR: no existe el archivo '{path}'", file=sys.stderr)
    print("Sugerencia: ejecuta primero ./scripts/profile.sh", file=sys.stderr)
    sys.exit(1)

if not os.path.isfile(path):
    print(f"ERROR: '{path}' no es un archivo regular", file=sys.stderr)
    sys.exit(1)

# ----------------------------------------------------------------------------
#  2. Lectura del CSV
# ----------------------------------------------------------------------------
try:
    with open(path) as f:
        data = list(csv.DictReader(f))
except OSError as e:
    print(f"ERROR: no se pudo leer '{path}': {e}", file=sys.stderr)
    sys.exit(1)

# ----------------------------------------------------------------------------
#  3. Validacion de contenido no vacio    [MEJORA 2]
# ----------------------------------------------------------------------------
if len(data) == 0:
    print(f"ERROR: '{path}' no contiene filas de datos", file=sys.stderr)
    sys.exit(1)

columns = ['io_ms', 'preprocess_ms', 'lk_scalar_ms', 'save_ms', 'total_ms']

missing = [c for c in columns if c not in data[0]]
if missing:
    print(f"ERROR: faltan columnas en el CSV: {missing}", file=sys.stderr)
    sys.exit(1)

# ----------------------------------------------------------------------------
#  3b. Filtrado de filas con valores no numericos    [ROBUSTEZ]
# ----------------------------------------------------------------------------
# Descarta filas donde alguna columna no sea un float valido. Esto protege
# contra CSVs mal formados (por ejemplo, cuando el parser de profile.sh
# captura lineas incidentales como "Pipeline total: X ms" y deja "ms" en
# lugar del valor numerico).
def is_number(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

clean_data = []
skipped = 0
for row in data:
    if all(is_number(row.get(c, "")) for c in columns):
        clean_data.append(row)
    else:
        skipped += 1

if skipped > 0:
    print(f"ADVERTENCIA: se omitieron {skipped} filas con valores no numericos",
          file=sys.stderr)

if len(clean_data) == 0:
    print(f"ERROR: no quedan filas validas en '{path}'", file=sys.stderr)
    print("Revisa el formato de salida del binario o el parser de profile.sh",
          file=sys.stderr)
    sys.exit(1)

data = clean_data

# ----------------------------------------------------------------------------
#  4. Encabezado de la tabla
# ----------------------------------------------------------------------------
print(f"Muestras: {len(data)}\n")
print(f"{'Etapa':<20} {'Media':>10} {'Mediana':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print("-" * 72)

# ----------------------------------------------------------------------------
#  5. Calculo de estadisticas
# ----------------------------------------------------------------------------
for col in columns:
    vals = [float(r[col]) for r in data]

    media   = statistics.mean(vals)
    mediana = statistics.median(vals)
    std     = statistics.stdev(vals) if len(vals) > 1 else 0.0
    minimo  = min(vals)
    maximo  = max(vals)

    print(f"{col:<20} {media:>10.3f} "
          f"{mediana:>10.3f} "
          f"{std:>10.3f} "
          f"{minimo:>10.3f} {maximo:>10.3f}")