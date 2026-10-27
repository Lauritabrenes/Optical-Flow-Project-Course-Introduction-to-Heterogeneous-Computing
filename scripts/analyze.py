#!/usr/bin/env python3
# ============================================================================
#  scripts/analyze.py
# ----------------------------------------------------------------------------
import csv
import os
import statistics
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "../results/profiling/times_scalar.csv"

if not os.path.exists(path) or not os.path.isfile(path):
    print(f"ERROR: no existe el archivo o ruta '{path}'", file=sys.stderr)
    sys.exit(1)

try:
    with open(path) as f:
        data = list(csv.DictReader(f))
except OSError as e:
    print(f"ERROR: no se pudo leer '{path}': {e}", file=sys.stderr)
    sys.exit(1)

if len(data) == 0:
    print(f"ERROR: '{path}' no contiene filas de datos", file=sys.stderr)
    sys.exit(1)

columns = ['io_ms', 'preprocess_ms', 'lk_compute_ms', 'save_ms', 'total_ms']

missing = [c for c in columns if c not in data[0]]
if missing:
    print(f"ERROR: faltan columnas en el CSV: {missing}", file=sys.stderr)
    sys.exit(1)

def is_number(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

clean_data = [row for row in data if all(is_number(row.get(c, "")) for c in columns)]

if len(clean_data) == 0:
    print(f"ERROR: no quedan filas validas en '{path}'", file=sys.stderr)
    sys.exit(1)

print(f"Muestras validas: {len(clean_data)} (Archivo: {os.path.basename(path)})\n")
print(f"{'Etapa':<20} {'Media':>10} {'Mediana':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print("-" * 72)

for col in columns:
    vals = [float(r[col]) for r in clean_data]
    media   = statistics.mean(vals)
    mediana = statistics.median(vals)
    std     = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"{col:<20} {media:>10.3f} {mediana:>10.3f} {std:>10.3f} {min(vals):>10.3f} {max(vals):>10.3f}")