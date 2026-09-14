"""Show whether the two long experiments are still running and how far along."""
import os
import re
import subprocess

LOGS = {"Expanded 100-attack study": "results/_log_expanded.txt",
        "Adaptive attack study": "results/_log_adaptive.txt"}
OUTS = {"Expanded 100-attack study": "results/raw_defenses_expanded.csv",
        "Adaptive attack study": "results/raw_adaptive.csv"}

try:
    ps = subprocess.check_output(["ps", "-W"], text=True, stderr=subprocess.DEVNULL)
    alive = ps.count("AI-Economics/.venv")
except Exception:
    alive = -1
print(f"project python processes running: "
      f"{alive if alive >= 0 else 'unknown (ps unavailable)'}\n")

for name, log in LOGS.items():
    print("=" * 68)
    print(f"  {name}")
    out = OUTS[name]
    if os.path.exists(out):
        try:
            import pandas as pd
            d = pd.read_csv(out)
            print(f"  saved rows: {len(d)}")
            if "defense" in d:
                print("  completed defenses: "
                      + ", ".join(sorted(d["defense"].unique())))
        except Exception as e:
            print("  (could not read csv:", e, ")")
    else:
        print("  no results file yet (saves after each defense completes)")
    if os.path.exists(log):
        txt = open(log, encoding="utf-8", errors="replace").read().replace("\r", "\n")
        lines = [l for l in txt.split("\n") if l.strip()]
        bars = [l for l in lines if "it/s" in l or "s/it" in l]
        if bars:
            print("  latest:", re.sub(r"\s+", " ", bars[-1])[:90])
        saved = [l for l in lines if l.startswith("[saved]") or l.startswith("[done]")]
        for s in saved[-3:]:
            print("  ", s[:90])
    print()
