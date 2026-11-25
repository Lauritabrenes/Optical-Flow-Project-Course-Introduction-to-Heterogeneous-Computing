import glob, time, csv, numpy as np
from pynq import Overlay, allocate
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
H, W, WIN = 240, 320, 5
IN_DTYPE = np.float32      # np.uint8 if f1/f2 are unsigned char in HLS
SCALE = 1/255.0            # 1.0 if the IP expects 0-255
REPS = 10
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
    det = Sxx*Syy - Sxy**2; ok = np.abs(det) > 1e-6
    u = np.where(ok, (-Syy*Sxt + Sxy*Syt) / np.where(ok, det, 1), 0)
    v = np.where(ok, ( Sxy*Sxt - Sxx*Syt) / np.where(ok, det, 1), 0)
    return u.astype(np.float32), v.astype(np.float32)

frames = sorted(glob.glob('frames/frame_*_320x240_crop.png'))
assert len(frames) >= 2, 'need at least 2 frames in ./frames'
from pynq import MMIO; ol = Overlay('lk_v0.bit'); ip = MMIO(0xA0000000, 0x10000)

b0 = allocate((H*W,), dtype=IN_DTYPE); b1 = allocate((H*W,), dtype=IN_DTYPE)
bu = allocate((H*W,), dtype=np.float32); bv = allocate((H*W,), dtype=np.float32)
for k, b in (('img0',b0),('img1',b1),('u',bu),('v',bv)):
    ip.write(REG[k], b.device_address & 0xFFFFFFFF)
    ip.write(REG[k]+4, b.device_address >> 32)
ip.write(REG_W, W); ip.write(REG_H, H)

rows = []
for i in range(len(frames)-1):
    I0, I1 = load(frames[i]), load(frames[i+1])
    assert I0.shape == (H, W), I0.shape
    b0[:] = (I0*SCALE).astype(IN_DTYPE).ravel(); b1[:] = (I1*SCALE).astype(IN_DTYPE).ravel()
    ts_tx, ts_ip, ts_tot = [], [], []
    for _ in range(REPS):
        t0 = time.perf_counter(); b0.sync_to_device(); b1.sync_to_device()
        t1 = time.perf_counter(); ip.write(0x00, 1)
        while (ip.read(0x00) & 0x2) == 0:
            if time.perf_counter() - t1 > TIMEOUT_S:
                raise SystemExit('TIMEOUT: IP never finished (.bit/.hwh mismatch or wrong registers)')
        t2 = time.perf_counter(); bu.sync_from_device(); bv.sync_from_device()
        t3 = time.perf_counter()
        ts_tx.append((t1-t0)+(t3-t2)); ts_ip.append(t2-t1); ts_tot.append(t3-t0)
    tc = time.perf_counter(); ru, rv = lk_ref(I0*SCALE, I1*SCALE); tc = time.perf_counter()-tc
    u, v = np.array(bu).reshape(H,W), np.array(bv).reshape(H,W)
    m = (slice(WIN,-WIN), slice(WIN,-WIN))
    eu, ev = np.abs(u-ru)[m], np.abs(v-rv)[m]
    row = dict(pair=f'{i+1}-{i+2}', ip_us=np.median(ts_ip)*1e6, xfer_us=np.median(ts_tx)*1e6,
               total_us=np.median(ts_tot)*1e6, cpu_ref_us=tc*1e6, clk_mhz=CLK_MHZ,
               max_abs_err_u=eu.max(), max_abs_err_v=ev.max(), mean_abs_err=(eu.mean()+ev.mean())/2)
    print(row); rows.append(row)
    np.save(f'flow_{i+1:06d}.npy', np.stack([u, v]))

with open('results_lk.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print('Saved results_lk.csv and flow_*.npy')
