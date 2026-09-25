import argparse
import os
import subprocess
import sys


def _banner(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def _run(module):
    """Run a python module as a subprocess so each step gets a clean state."""
    cmd = [sys.executable, "-m", module]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train",   action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    os.makedirs("task4/results",     exist_ok=True)
    os.makedirs("task4/checkpoints", exist_ok=True)
    os.makedirs("task4/cache",       exist_ok=True)

    if not args.skip_train:
        _banner("Step 1: Train Vanilla")
        from task4.methods.vanilla import train
        train("vanilla")

        _banner("Step 2: Train GCSC")
        train("gcsc")

    else:
        print("[skip] Training (--skip-train)")

    if not args.skip_extract:
        _banner("Step 3 & 4: Extract Vanilla + GCSC outputs")
        for method in ["vanilla", "gcsc"]:
            import subprocess, sys
            subprocess.run(
                [sys.executable, "-m", "task4.extract_outputs",
                 "--method", method], check=True)
    else:
        print("[skip] Extraction (--skip-extract)")

    if not args.skip_train:
        _banner("Step 5: Train PROSER")
        from task4.methods.proser import train as train_proser
        train_proser()
    
    if not args.skip_extract:
        _banner("Step 6: Extract PROSER outputs")
        import subprocess, sys
        subprocess.run(
            [sys.executable, "-m", "task4.extract_outputs",
             "--method", "proser"], check=True)

    _banner("Step 7: OSR Evaluation")
    from task4.evaluate_osr import main as eval_main
    eval_main()

    print("\n" + "="*60)
    print("  Task 4 complete.  Results in task4/results/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
