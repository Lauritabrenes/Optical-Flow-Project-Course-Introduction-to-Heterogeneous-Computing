#!/bin/bash
# usage: sample_power.sh <output_file> <seconds|0=until killed>
OUT="${1}"; DUR=${2:-0}; : > "$OUT"; T0=$(date +%s)
P=/sys/class/hwmon/hwmon2/power1_input
while :; do
  echo "$(date +%s.%N) SOM total power : $(( $(cat $P) / 1000 )) mW" >> "$OUT"
  sleep 0.2
  [ "$DUR" -gt 0 ] && [ $(( $(date +%s) - T0 )) -ge "$DUR" ] && break
done
