"""Regenerate every figure and printed number in the README from the files in results/."""
import confidence_sweep
import enhancement_ablation
import robustness
import training_curves

for module in (training_curves, confidence_sweep, enhancement_ablation, robustness):
    print(f"\n== {module.__name__} ==")
    module.main()
