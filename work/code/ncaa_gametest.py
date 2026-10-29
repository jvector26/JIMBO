import numpy as np, pandas as pd
exec(open("/home/claude/code/young_prior_fit.py").read().split("MODELS =")[0])
R = ["rk", "rk_lp", "rk_age"]
MODELS = {"pick+age (current new)": BASE + R, "pick+age+college": BASE + R + ["rk_cs", "rk_nocol"],
          "college only (+nocol)": BASE + ["rk", "rk_cs", "rk_nocol"], "old 3 bins": BASE + ["rk_1_3", "rk_4_10", "rk_11_30"]}
r0 = loso(MODELS["old 3 bins"])
for k, c in MODELS.items():
    b, t = fit(A, c); r = loso(c)
    print(f"{k:24s} LOSO {r:.4f} ({r0-r:+.4f})  test21-25 {split(c):.4f} | " + ", ".join(f"{n} {v/b[1]:+.2f} ({tt:.1f})" for n, v, tt in zip(c, b[1:], t[1:]) if n.startswith("rk")))
    for nm, d in [("15-19", A[A.season <= 2019]), ("20-25", A[A.season >= 2020])]:
        if "rk_cs" in c: b2, t2 = fit(d, c); print(f"     {nm}: rk_cs {b2[c.index('rk_cs')+1]/b2[1]:+.2f} (t {t2[c.index('rk_cs')+1]:.1f})")
