"""run_lk_prof.py - Perfilado de lk_hls_v0 en el KV260 (>100 muestras por etapa)

Basado en run_lk.py. Cambios:
  * REPS configurable (por defecto 300) + WARMUP corridas descartadas.
  * Una fila POR CORRIDA en times_fpga_v0.csv (no promedios).
  * Tiempos por etapa: load_ms (lectura PNG + escala), h2d_ms, kernel_ms,
    d2h_ms, total_ms.
  * Estadisticas (media, mediana, std, min, max) en stats_fpga_v0.csv
    y en pantalla, mismo formato que scripts/analyze.py.
  * WIN = 7 en la referencia (el kernel usa ventana 7x7, LK_RADIUS=3).
    Con WIN=5 el error contra la referencia salia ~24.

Uso:  bash run_prof.sh            (lanza tambien platformstats)
      o: python3 run_lk_prof.py [REPS]
"""
import glob, sys, time, csv, numpy as np
from pynq import Overlay, allocate, MMIO
try:
    from PIL import Image
    load = lambda p: np.asarray(Image.open(p).convert('L'), dtype=np.float32)
except ImportError:
    import cv2
    load = lambda p: cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32)

# ===== CONFIG (from lk_v0.hwh) =====
IP_NAME = 'lk_hls_v0_0'
REG = {'img0': 0x10, 'img1': 0x1c, 'u': 0x28, 'v': 0x34}   # f1, f2, u, v
REG_W, REG_H = 0x40, 0x48
H, W = 240, 320
WIN = 7                    # ventana del kernel (2*LK_RADIUS+1)
MARGIN = WIN // 2 + 1      # LK_MARGIN del kernel
IN_DTYPE = np.float32
SCALE = 1/255.0
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
WARMUP = 3
CLK_MHZ = 142.86
TIMEOUT_S = 5
# ===================================

def lk_ref(I0, I1, win=WIN):
    Ix = np.zeros_like(I0); Iy = np.zeros_like(I0)
    Ix[:,1:-1] = (I0[:,2:] - I0[:,:-2]) / 2
    Iy[1:-1,:] = (I0[2:,:] - I0[:-2,:]) / 2
    It = I1 - I0
    def box(a):
        r = win//2; p = np.pad(a, r); c = p.cumsum(0).cumsum(1)
        c = np.pad(c, ((1,0),(1,0)))
        return c[win:,win:] - c[:-win,win:] - c[win:,:-win] + c[:-win,:-win]
    Sxx, Syy, Sxy = box(Ix*Ix), box(Iy*Iy), box(Ix*Iy)
    Sxt, Syt = box(Ix*It), box(Iy*It)
    det = Sxx*Syy - Sxy**2; ok = det > 1e-6
    u = np.where(ok, (-Syy*Sxt + Sxy*Syt) / np.where(ok, det, 1), 0)
    v = np.where(ok, ( Sxy*Sxt - Sxx*Syt) / np.where(ok, det, 1), 0)
    return u.astype(np.float32), v.astype(np.float32)

frames = sorted(glob.glob('frames/frame_*_320x240_crop.png'))
assert len(frames) >= 2, 'need at least 2 frames in ./frames'
F0, F1 = frames[0], frames[1]

ol = Overlay('lk_v0.bit'); ip = MMIO(0xA0000000, 0x10000)
b0 = allocate((H*W,), dtype=IN_DTYPE); b1 = allocate((H*W,), dtype=IN_DTYPE)
bu = allocate((H*W,), dtype=np.float32); bv = allocate((H*W,), dtype=np.float32)
for k, b in (('img0',b0),('img1',b1),('u',bu),('v',bv)):
    ip.write(REG[k], b.device_address & 0xFFFFFFFF)
    ip.write(REG[k]+4, b.device_address >> 32)
ip.write(REG_W, W); ip.write(REG_H, H)

ms = lambda a, b: (b - a) * 1e3
rows = []
print(f'Perfilando lk_hls_v0: {WARMUP} warmup + {REPS} corridas ({W}x{H}) ...')
t_start = time.time()
for k in range(WARMUP + REPS):
    t0 = time.perf_counter()
    I0, I1 = load(F0), load(F1)
    b0[:] = (I0*SCALE).astype(IN_DTYPE).ravel(); b1[:] = (I1*SCALE).astype(IN_DTYPE).ravel()
    t1 = time.perf_counter(); b0.sync_to_device(); b1.sync_to_device()
    t2 = time.perf_counter(); ip.write(0x00, 1)
    while (ip.read(0x00) & 0x2) == 0:
        if time.perf_counter() - t2 > TIMEOUT_S:
            raise SystemExit('TIMEOUT: IP never finished (.bit/.hwh mismatch or wrong registers)')
    t3 = time.perf_counter(); bu.sync_from_device(); bv.sync_from_device()
    t4 = time.perf_counter()
    if k >= WARMUP:
        rows.append([k - WARMUP + 1, ms(t0,t1), ms(t1,t2), ms(t2,t3), ms(t3,t4), ms(t0,t4)])
t_end = time.time()

cols = ['run', 'load_ms', 'h2d_ms', 'kernel_ms', 'd2h_ms', 'total_ms']
with open('times_fpga_v0.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(cols); w.writerows(rows)

# --- estadisticas ---
a = np.array(rows)[:, 1:]
print(f'\n================ ANALISIS ESTADISTICO ================')
print(f'Muestras validas: {len(rows)} (Archivo: times_fpga_v0.csv)\n')
print(f"{'Etapa':<16}{'Media':>11}{'Mediana':>11}{'Std':>11}{'Min':>11}{'Max':>11}")
print('-' * 71)
stats = []
for name, c in zip(cols[1:], a.T):
    s = [name, c.mean(), np.median(c), c.std(ddof=1), c.min(), c.max()]
    stats.append(s)
    print(f'{name:<16}' + ''.join(f'{x:>11.3f}' for x in s[1:]))
print('=' * 54)
with open('stats_fpga_v0.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['stage','mean_ms','median_ms','std_ms','min_ms','max_ms']); w.writerows(stats)

# --- correctitud (una vez) contra la referencia NumPy ---
ru, rv = lk_ref(I0*SCALE, I1*SCALE)
u, v = np.array(bu).reshape(H,W), np.array(bv).reshape(H,W)
m = (slice(MARGIN,-MARGIN), slice(MARGIN,-MARGIN))
eu, ev = np.abs(u-ru)[m], np.abs(v-rv)[m]
kern = a[:, 2]
summary = dict(frames=f'{F0} {F1}', reps=REPS, clk_mhz=CLK_MHZ,
               kernel_mean_ms=kern.mean(), kernel_std_ms=kern.std(ddof=1),
               cycles_per_pixel=kern.mean()*1e-3*CLK_MHZ*1e6/(W*H),
               fps_kernel=1e3/kern.mean(),
               max_abs_err_u=eu.max(), max_abs_err_v=ev.max(),
               mean_abs_err=(eu.mean()+ev.mean())/2,
               run_start_epoch=t_start, run_end_epoch=t_end)
print(summary)
with open('summary_fpga_v0.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=summary.keys()); w.writeheader(); w.writerow(summary)
np.save('flow_000001.npy', np.stack([u, v]))
print('Saved times_fpga_v0.csv, stats_fpga_v0.csv, summary_fpga_v0.csv, flow_000001.npy')
