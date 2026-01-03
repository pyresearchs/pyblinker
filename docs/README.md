# pyblinker Documentation

This directory contains detailed documentation for the `pyblinker` package. The files are ordered sequentially to guide you through the pipeline.

## Main Documentation

1.  [**Overview**](00_overview.md)
    *   High-level purpose, pipeline steps, and design goals.
2.  [**Blink Region and Candidates**](01_blink_region_and_candidates.md)
    *   How to detect candidate blink regions from EEG, EOG, or EAR.
3.  [**Blink Segmentation and Refinement**](02_blink_segmentation_refinement.md)
    *   Refining start/end points using zero-crossing, thresholding, etc.
4.  [**Epoch-Based Pipeline**](03_epoch_based_pipeline.md)
    *   Why and how to process data in epochs.
5.  [**Blink Metrics and Features**](04_blink_metrics_and_features.md)
    *   Comprehensive list of computed features (Kinematics, Energy, EAR).
6.  [**Reporting and QC**](05_reporting_and_qc.md)
    *   Generating HTML reports for quality control.

## Appendices

*   [**Utils Inventory**](90_utils_inventory.md)
    *   List of utility functions and their signatures.
*   [**Migration Mapping**](91_utils_migration_mapping.md)
    *   Mapping of legacy modules to the new structure.
*   [**Test Data Provenance**](92_test_data_provenance.md)
    *   Origin of the test files included in the repository.
