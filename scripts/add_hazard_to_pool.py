import pickle, sys, time
from lunarsim.core.terrain.hazard import attach_hazard
for path in sys.argv[1:]:
    T = pickle.load(open(path, "rb")); t0 = time.time()
    fr = []
    for t in T:
        attach_hazard(t); fr.append(float((t.haz_dist <= 0).mean()))
    pickle.dump(T, open(path, "wb"))
    import numpy as np
    print(f"{path}: {len(T)} tiles, safe-cell fraction mean {np.mean(fr):.3f} (p10 {np.percentile(fr,10):.3f}, "
          f"p90 {np.percentile(fr,90):.3f}), median dist-to-safe at cell {np.median([float(np.median(t.haz_dist)) for t in T]):.0f} m, {time.time()-t0:.0f}s")
