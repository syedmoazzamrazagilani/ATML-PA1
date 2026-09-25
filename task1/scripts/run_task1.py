import argparse
import sys
import os


def _banner(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Run Task 1 experiments")
    parser.add_argument("--skip-data",  action="store_true",
                        help="Skip data download and split creation")
    parser.add_argument("--skip-cue",   action="store_true",
                        help="Skip cue-conflict image generation")
    parser.add_argument("--skip-repr",  action="store_true",
                        help="Skip representation analysis (t-SNE)")
    args = parser.parse_args()

    if not args.skip_data:
        _banner("Step 0: Creating data splits")
        from task1.data.make_subset import prepare_splits
        prepare_splits()
    else:
        print("[Step 0] Skipping data split (--skip-data)")

    _banner("Step 1: Training linear heads & clean baseline")
    from task1.models.train_heads import train_heads
    train_heads()

    if not args.skip_cue:
        _banner("Step 2: Generating cue-conflict images")
        from task1.data.make_cue_conflicts import generate_conflicts
        generate_conflicts()
    else:
        print("[Step 2] Skipping cue-conflict generation (--skip-cue)")

    _banner("Step 3: Color bias (grayscale + hue rotation)")
    from task1.analysis.evaluate_color_bias import evaluate_color_bias
    evaluate_color_bias()

    _banner("Step 4: Cue-conflict shape-bias evaluation")
    from task1.analysis.evaluate_cue_conflicts import evaluate_cue_conflicts
    evaluate_cue_conflicts()

    _banner("Step 5: Translation invariance")
    from task1.analysis.evaluate_translation import evaluate_translation
    evaluate_translation()

    _banner("Step 6: Patch structure (4×4 shuffle)")
    from task1.analysis.evaluate_patch_shuffle import evaluate_patch_shuffle
    evaluate_patch_shuffle()

    if not args.skip_repr:
        _banner("Step 7: Representation analysis (cosine stability + t-SNE)")
        from task1.analysis.run_representation_analysis import main as repr_main
        repr_main()
    else:
        print("[Step 7] Skipping representation analysis (--skip-repr)")

    print("\n" + "="*60)
    print("  Task 1 complete.  Results in task1/results/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
