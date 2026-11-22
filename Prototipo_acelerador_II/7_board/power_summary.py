import re, statistics as st
def read(fn):
    return [float(m.group(1)) for l in open(fn) if (m := re.search(r'SOM total power\s*:\s*([\d.]+)\s*mW', l))]
idle, run = read('platformstats_idle.txt'), read('platformstats_run.txt')
for n, v in (('idle', idle), ('run', run)):
    print(f"{n:5s} samples={len(v):3d}  mean={st.mean(v):.1f} mW  max={max(v):.1f} mW" if v else f"{n}: no samples")
if idle and run:
    d = st.mean(run) - st.mean(idle); k = 22.51e-3
    print(f"dynamic power      = {d:.1f} mW")
    print(f"energy/frame total = {st.mean(run)*k:.2f} mJ")
    print(f"energy/frame dyn.  = {d*k:.2f} mJ")
