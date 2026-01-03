# Agent Protocol: Documentation Maintenance

**CRITICAL INSTRUCTION FOR AGENTS AND CONTRIBUTORS**

To maintain the high quality of this repository's documentation, you **MUST** follow this protocol whenever you refactor code or add a new feature.

## The Rule

> **"Code changes are not complete until the documentation reflects them."**

For every significant change (new feature, bug fix, architectural refactor), you are required to update the relevant Markdown file(s) in the `docs/` directory.

## What to Update

When modifying the codebase, identify which stage of the pipeline your change affects and update the corresponding document:

*   **Detection/Candidates**: Update `docs/01_blink_region_and_candidates.md`
*   **Refinement**: Update `docs/02_blink_segmentation_refinement.md`
*   **Epoching**: Update `docs/03_epoch_based_pipeline.md`
*   **Features**: Update `docs/04_blink_metrics_and_features.md`
*   **Reporting**: Update `docs/05_reporting_and_qc.md`
*   **Legacy Porting**: Update `docs/06_matlab_migration_and_replication.md`

## Required Content

Your documentation update must explicitly cover these three points:

1.  **The Feature/Change**: Briefly explain what was added or changed.
2.  **Related Code**: List the **exact file path(s)** of the Python scripts implementing the logic.
3.  **Verification (Tutorials & Tests)**:
    *   **Tutorials**: If you added a new tutorial script or if an existing one demonstrates the feature, link to it (e.g., `tutorial/new_feature_demo.py`).
    *   **Unit Tests**: Cite the specific unit test file that validates this logic (e.g., `test/my_new_feature/test_logic.py`).

## Example Update

If you add a new "Slope Asymmetry" feature to the metrics module:

1.  Open `docs/04_blink_metrics_and_features.md`.
2.  Add a new entry under the relevant section (e.g., "Kinematics").
3.  Append the new files to the "Related Code" section.
4.  Add the new test to the "Unit Tests" section.

```markdown
### Slope Asymmetry
*   **`slope_asymmetry`**: The ratio of the closing slope to the opening slope.

...

## Related Code
*   `pyblinker/blink_features/kinematics/new_slope_feature.py`

## Unit Tests
*   `test/blink_features/kinematics/test_slope_asymmetry.py`: Validates the ratio calculation against synthetic triangular waves.
```

**Do not skip this step.** Accurate documentation is as important as passing tests.
