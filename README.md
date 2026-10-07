# Portable-Bridge U-Net — restructured code + manuscript audit

This folder separates the **code path actually used by the supplied notebooks** from the many experimental/legacy variants in the original archive.

## Core correspondence

- `legacy_reference/model.py` → baseline U-Net used for the manuscript parameter comparison.
- `legacy_reference/prop_model.py` → Portable-Bridge architecture.
- `legacy_reference/data.py` + `utils.py::Data.load_data` → preprocessing and 70/15/15 image-level split.
- `legacy_reference/post_processing.py` → hole filling + largest connected component.
- `src/portable_bridge_unet/` → cleaned implementation preserving the architecture, but removing accidental mutable-list behavior and separating responsibilities.

## Important audit findings

1. **Architecture identity is strongly confirmed by exact parameter counts.**
   The legacy `model.py` gives 1,213,953 params and `prop_model.py` gives 2,360,321 params for `[32,64,128]`, exactly matching manuscript Table 1.
2. **The original proposed-model function mutates `num_filters` using `reverse()` in place.** Repeated calls with the same list can silently alternate the encoder order. The restructured code fixes this.
3. **Code uses RGB/3-channel input**, while the manuscript describes a single-channel input.
4. **Actual notebook split for 7,365 images is 5,157 / 1,104 / 1,104**, not 5,155 / 1,104 / 1,104.
5. **Split is image-wise**, so slices from the same patient can potentially cross train/validation/test unless filenames/dataset structure prevent it externally.
6. **Post-processing code is hole filling + largest connected component.** The manuscript wording about replacing missing pixels using an average contour value is not implemented.
7. Some legacy prediction scripts compute averages only on selected "good" cases and can append very-high-IoU cases twice. Those averages should not be used as whole-test-set metrics.
8. The supplied `Proposed_Inner_PostProcessed.csv` values for `IM-0001-2415` and `IM-0001-7127` reproduce the manuscript's post-processing examples almost exactly, providing direct provenance evidence.

Run `python audit/verify_parameter_counts.py` without TensorFlow to verify the two parameter counts.
