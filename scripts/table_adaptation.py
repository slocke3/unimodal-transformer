import numpy as np, glob, re
from src.maps import iterate_asym, asym_map_vec

NB = 64
def clamp(R, p_probe, p_edge, x):
    return float(np.sqrt(np.mean((asym_map_vec(x, R, p_probe)
                                - asym_map_vec(x, R, p_edge))**2)))

HDR = ("%6s %9s %8s %9s %10s %10s %9s %12s"
       % ("band w","edge a","probe a","dx_c bins","model","clamp","floor","adaptation"))
def row(w, edge, probe, dxc, m, c, f):
    return ("%6g %9.2f %8.2f %9.1f %10.5f %10.5f %9.5f %11.0f%%"
            % (w, edge, probe, dxc*NB, m, c, f, 100*(c-m)/max(c-f,1e-12)))

# ---- token models: pick, per run, the probe closest to dx_c = 0.1 ----------
p = np.load("figures_asym2/return_map_probe.npz")
runs = sorted({re.sub(r"_(alpha|d_true|d_train|floor|band_lo|x_.*|e_.*)$","",k)
               for k in p.files if k.endswith("_band_lo")})
print("TOKEN input  (asym family, R=1)")
print(HDR)
for r in runs:
    lo = float(p[f"{r}_band_lo"]); al = p[f"{r}_alpha"]
    dt, fl = p[f"{r}_d_true"], p[f"{r}_floor"]
    cand = [(abs((lo-a_)/2 - 0.1), a_, d_, f_)
            for a_, d_, f_ in zip(al, dt, fl) if a_ < lo - 1e-9]
    if not cand: continue
    _, a_, d_, f_ = min(cand)
    x = p[f"{r}_x_{a_:g}"]
    w = round(1.0 - lo, 2)
    print(row(w, lo, a_, (lo-a_)/2, d_, clamp(1.0, float(a_), lo, x), f_))

# ---- continuous models: same columns, probe at fixed d = 0.2 --------------
def clamp_grid(p_probe, p_edge, R_grid, L=50, traj=150, n=30, seed=3):
    rng = np.random.default_rng(seed); out = []
    for R in R_grid:
        x = np.concatenate([iterate_asym(rng.uniform(.05,.95), R, p_probe, traj)[L-1:-1]
                            for _ in range(n)])
        out.append(clamp(R, p_probe, p_edge, x))
    return float(np.mean(out))

for om in ("scalar", "bins"):
    print(f"\nCONTINUOUS input, {om} output  (asym family, R>=0.25)")
    print(HDR)
    for d in sorted(glob.glob(f"runs_cont/asym_{om}_w*_seed0"),
                    key=lambda s: float(re.search(r"_w([\d.]+)_", s).group(1))):
        z = np.load(d+"/eval_cont.npz"); w = float(z["band_width"])
        keep = z["R_grid"] >= 0.25; hm = z["heldout_mask"]
        edge, probe = float(z["band_lo"]), float(z["p_grid"][hm][0])
        m = float(np.nanmean(z["implied_rms"][hm][:, keep]))
        f = float(np.nanmean(z["binning_floor"][hm][:, keep]))
        print(row(w, edge, probe, (edge-probe)/2,
                  m, clamp_grid(probe, edge, z["R_grid"][keep][::6]), f))
