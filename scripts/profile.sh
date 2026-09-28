#!/bin/bash
# ============================================================================
#  scripts/profile.sh
# ----------------------------------------------------------------------------
#  Perfilado del pipeline de Optical Flow (Escalar + NEON).
#
#  Ejecuta el binario `bench_main` N veces con el mismo par de frames y
#  extrae los tiempos de cada etapa del pipeline, acumulandolos en un CSV.
#
#  Formato esperado en la salida del binario:
#      [1] IO          <tiempo> ms
#      [2] PREPROCESS  <tiempo> ms
#      [3] LK_SCALAR   <tiempo> ms
#      [4] LK_NEON     <tiempo> ms
#      [5] SAVE        <tiempo> ms
#      TOTAL           <tiempo> ms
#
#  Uso:
#      ./profile.sh [BIN] [FRAME1] [FRAME2] [N] [OUT]
# ============================================================================

set -e

# ----------------------------------------------------------------------------
#  1. Parametros de entrada
# ----------------------------------------------------------------------------
BIN="${1:-../build/bench_main}"
FRAME1="${2:-../data/frames/frame_000001.png}"
FRAME2="${3:-../data/frames/frame_000002.png}"
N="${4:-100}"
OUT="${5:-../results/profiling/times.csv}"

# ----------------------------------------------------------------------------
#  2. Verificacion de prerrequisitos
# ----------------------------------------------------------------------------
if [[ ! -x "$BIN" ]]; then
    echo "ERROR: binario no encontrado o no ejecutable: $BIN" >&2
    exit 1
fi
[[ -f "$FRAME1" ]] || { echo "ERROR: frame1 no encontrado: $FRAME1" >&2; exit 1; }
[[ -f "$FRAME2" ]] || { echo "ERROR: frame2 no encontrado: $FRAME2" >&2; exit 1; }

# ----------------------------------------------------------------------------
#  3. Preparacion del directorio y archivo de salida
# ----------------------------------------------------------------------------
mkdir -p "$(dirname "$OUT")"
# Se añade la columna lk_neon_ms al encabezado del CSV
echo "run,io_ms,preprocess_ms,lk_scalar_ms,lk_neon_ms,save_ms,total_ms" > "$OUT"

# ----------------------------------------------------------------------------
#  4. Bucle de perfilado
# ----------------------------------------------------------------------------
for i in $(seq 1 "$N"); do

    # Filtro estricto: solo lineas que EMPIEZAN con [1], [2], [3], [4], [5] o TOTAL.
    LINE=$("$BIN" "$FRAME1" "$FRAME2" 2>/dev/null | \
        grep -E "^(\[[1-5]\]|TOTAL)" | \
        awk '{for (j=1; j<=NF; j++) if ($j ~ /^-?[0-9]+(\.[0-9]+)?$/) v=$j; printf "%s,", v}')

    if [[ -z "$LINE" ]]; then
        echo "WARN: run $i sin salida valida del binario (se omite)" >&2
        continue
    fi

    # Verifica que tengamos exactamente 6 valores antes de escribir
    NFIELDS=$(echo "$LINE" | awk -F',' '{print NF-1}')  # -1 por la coma final
    if [[ "$NFIELDS" -ne 6 ]]; then
        echo "WARN: run $i con$NFIELDS campos (esperado 6), se omite" >&2
        continue
    fi

    echo "$i,${LINE\%,}" >> "$OUT"
done

# ----------------------------------------------------------------------------
#  5. Resumen final
# ----------------------------------------------------------------------------
ROWS=$(( $(wc -l < "$OUT") - 1 ))
echo "Perfilado completo: $OUT ($ROWS/$N muestras validas)"