#!/bin/bash
# run_prof.sh - Perfilado de lk_hls_v0 + platformstats en reposo y durante la ejecucion
# Uso (en la Kria):  bash run_prof.sh [REPS]      (por defecto 300)
cd /home/el5859/deploy
REPS=${1:-300}

echo "== 1. platformstats en reposo (10 s)"
sudo bash -c 'for i in $(seq 1 10); do echo "### $(date +%s.%N)"; platformstats; sleep 1; done' > platformstats_idle.txt 2>&1

echo "== 2. platformstats durante la ejecucion (cada 0.5 s)"
sudo bash -c 'while true; do echo "### $(date +%s.%N)"; platformstats; sleep 0.5; done' > platformstats_run.txt 2>&1 &
PS_PID=$!
sleep 1

echo "== 3. Perfilado ($REPS corridas)"
sudo bash -c "source /etc/profile.d/pynq_venv.sh; source /etc/profile.d/xrt_setup.sh 2>/dev/null; cd /home/el5859/deploy && python3 run_lk_prof.py $REPS" | tee prof_fpga_v0.log

sudo pkill -P $PS_PID 2>/dev/null; sudo kill $PS_PID 2>/dev/null

echo "== 4. Resumen de potencia (SOM total power, mW)"
for f in platformstats_idle.txt platformstats_run.txt; do
  awk -v f="$f" '/SOM total power/ {gsub(/[^0-9]/,"",$NF); s+=$NF; n++; if($NF>mx)mx=$NF}
                 END {if(n) printf "%-24s muestras=%d  media=%.0f mW  max=%d mW\n", f, n, s/n, mx}' "$f"
done
echo "Listo: times_fpga_v0.csv stats_fpga_v0.csv summary_fpga_v0.csv platformstats_idle.txt platformstats_run.txt prof_fpga_v0.log"
