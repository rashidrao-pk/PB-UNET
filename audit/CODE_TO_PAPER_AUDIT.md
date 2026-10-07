# Code-to-paper correspondence audit

## Confidence summary

**Conclusion:** the supplied code is clearly the code lineage behind the manuscript, but the manuscript is **not a fully faithful description of the executable pipeline**. The architectural core and post-processing examples match strongly; several protocol, input-shape, and evaluation details need correction before publication.

## Direct matches

| Manuscript claim/artifact | Supplied code/evidence | Status |
|---|---|---|
| Baseline U-Net total params = 1,213,953 | `model.py`, filters 32/64/128 | Exact match |
| Proposed total params = 2,360,321 | `prop_model.py`, filters 32/64/128 | Exact match |
| Encoder → portable bridge → decoder | `prop_model.build_model()` | Match |
| Double 3×3 Conv + BN + ReLU blocks | `conv_block()` | Match |
| BCE loss | notebooks compile with `binary_crossentropy` | Match |
| 256×256 resize | `data.py` | Match |
| 70/15/15 concept | split code | Match conceptually |
| Post-process connected regions / retain largest | `post_processing.py` | Match |
| Table-9 examples (2415, 7127) | `Result/Proposed_Inner_PostProcessed.csv` | Near-exact numeric match |

## Mismatches / risks

### Input channels
Manuscript: single-channel input. Code: `Input((256,256,3))` and `cv2.IMREAD_COLOR`.

### Split counts
Notebook output: `5157 1104`, with test size 1104. For 7,365 total this is 5,157/1,104/1,104. Manuscript states 5,155/1,104/1,104.

### Patient leakage risk
The supplied split is random image/slice-level splitting. No patient/group-aware split is implemented.

### Mutable filter-order bug
`prop_model.portable_bridge()` calls `num_filters.reverse()` and mutates the caller's list. Reusing the same list across multiple `get_model()` calls can produce alternating architectures. Paper Table 1 corresponds specifically to a fresh `[32,64,128]` ordering.

### Batch-size inconsistency in notebooks
`data.tf_dataset()` defaults to batch 8. Some notebook cells change a separate `batch` variable without passing it into `tf_dataset`, so the nominal and actual batch size can differ. The later loop using batch=8 happens to align.

### Post-processing description
Code performs morphological reconstruction (hole fill) and largest connected component selection. It does not compute replacement pixels from an "average contour value".

### Evaluation-selection bias in legacy scripts
`predict_proposed.py` appends metrics only when `iou_pred_1 > 0.60`; cases above 0.95 are appended again in a second branch. Therefore its printed averages are not unbiased whole-test-set averages.

### Selected-image tables
Several manuscript tables list a small number of images even though the test set has 1,104 samples. These are example/best-case tables, not a full-test summary, unless separate evidence is supplied.

## What should be used for a new clean rerun

- fresh immutable filter tuples;
- explicit batch size passed to the dataset pipeline;
- patient-level split if subject IDs are available;
- metrics over every test sample, reported as mean ± SD (and ideally median/IQR);
- Dice, IoU, HD95, precision, recall;
- post-processing evaluated as a true ablation over the same complete test set.
