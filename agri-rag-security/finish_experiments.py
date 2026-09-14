"""Run (or resume) the two remaining experiments, one after the other.

Safe to re-run: both experiments skip model x defense combinations that are
already complete in their result CSV, so an interrupted run picks up where it
stopped. Nothing is duplicated and nothing is lost.

Usage (from this folder, with the venv active):
    python finish_experiments.py
"""
import subprocess
import sys

STEPS = [
    ("Expanded 100-attack defense study (6 defenses, gpt-4o-mini)",
     [sys.executable, "-u", "-m", "src.run_defenses",
      "--models", "openai/gpt-4o-mini", "--sets", "indirect",
      "--out", "raw_defenses_expanded.csv"]),
    ("Adaptive indirect-injection study (3 rounds, gpt-4o-mini)",
     [sys.executable, "-u", "-m", "src.run_adaptive",
      "--rounds", "3", "--target", "openai/gpt-4o-mini",
      "--defenses", "no_defense", "spotlight_output"]),
]

for i, (name, cmd) in enumerate(STEPS, 1):
    print("\n" + "=" * 72)
    print(f"  STEP {i}/{len(STEPS)}: {name}")
    print("=" * 72, flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"\n[!] step {i} exited with code {rc}.")
        print("    Just run this script again - it resumes from where it stopped.")
        sys.exit(rc)

print("\n" + "=" * 72)
print("  BOTH EXPERIMENTS COMPLETE")
print("  results/raw_defenses_expanded.csv")
print("  results/raw_adaptive.csv")
print("=" * 72)
