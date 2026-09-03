import numpy as np, glob
from src.maps import iterate_family, family_map_vec

def clamp_rms(family, R_grid, p_probe, p_edge, L=50, traj=150, n=30, seed=3):
    """RMS between the TRUE map at the probe and the BAND-EDGE map, on the
    probe's own orbit points. This is exactly what a model that ignored the
    context and applied its nearest trained map would score."""
    rng = np.random.default_rng(seed); out = []
    for R in R_grid:
        xs = []
        for _ in range(n):
            t = iterate_family(rng.uniform(.05,.95), R, p_probe, traj, family)
            xs.append(t[L-1:-1])
        x = np.concatenate(xs)
        out.append(np.sqrt(np.mean((family_map_vec(x, R, p_probe, family)
                                  - family_map_vec(x, R, p_edge,  family))**2)))
    return float(np.mean(out))

print("Is there ANY adaptation to the shift, or does the model just apply its")
print("nearest trained map?  'clamp' = RMS you would score by using the band-edge")
print("map. Model below clamp = it moved toward the true out-of-band map.\n")
print("%8s %8s %6s %11s %10s %9s %s"
      % ("family","output","w","held-out","clamp","floor","adaptation"))
for fam in ("asym","tilted"):
    for om in ("bins","scalar"):
        for d in sorted(glob.glob(f"runs_cont/{fam}_{om}_w*_seed0")):
            z = np.load(d+"/eval_cont.npz"); w = float(z["band_width"])
            keep = z["R_grid"] >= 0.25; R = z["R_grid"][keep]
            hm = z["heldout_mask"]
            if not hm.any(): continue
            obs = float(np.nanmean(z["implied_rms"][hm][:, keep]))
            flo = float(np.nanmean(z["binning_floor"][hm][:, keep]))
            pr = float(z["p_grid"][hm][0])
            edge = float(z["band_lo"]) if pr < float(z["band_lo"]) else float(z["band_hi"])
            cl = clamp_rms(fam, R[::6], pr, edge)
            frac = 100*(cl-obs)/max(cl-flo, 1e-12)
            print("%8s %8s %6g %11.5f %10.5f %9.5f %6.0f%% of the way from clamp to floor"
                  % (fam, om, w, obs, cl, flo, frac))
        print()
