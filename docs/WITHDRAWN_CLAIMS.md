# Claims that earlier drafts made and the evidence does not support

Each was checked and found wrong, so none may appear in the manuscript.
`tools/check_withdrawn.py` runs against the built `.docx` and fails the build if
one reappears; it is the reason this list is a file and not a habit.

| Claim | Why it is wrong |
|---|---|
| The method improves AMOTP / box localisation | Our tracker and Poly-MOT submit the matched detection box verbatim (346,010 of 346,010 identical), so there AMOTP can only move through *which* boxes are submitted, never through better geometry. MCTrack post-processes its trajectories, so its AMOTP does move (-0.0026), but the interval reaches +0.0001 and the improvement is not claimed. |
| The grid is the only source of motion evidence, or one that "does not wait for convergence" | The grid covers 2.6% of a 100 m x 100 m area and its velocity is itself accumulated over sweeps. |
| No metric gets worse | Track initialisation delay, small classes and the overall one-identity share do get worse. |
| Identity switches are 2.1% of the error | The decomposition gives 1.8-4.3%, and it moves with the operating point. |
| The identity gain is independent of the tracker | On Poly-MOT the identity-switch interval contains zero (-26 [-128, +57]). It excludes zero on MCTrack (-936 [-1226, -688]), but that baseline is far more damaged, so the two are not comparable. |
| Any number from the 15-threshold evaluator | Everything is scored with the official 40 recall thresholds of devkit 1.1.11. |
| KITTI Car sAMOTA rises by 0.04 | True only for the released PointRCNN detections. With PV-RCNN it is +0.0023 and with Voxel R-CNN it is negative, so the detector condition must appear in the same sentence. |
| The sAMOTA computed without its two lowest recall points | That cut was chosen after seeing the results. It is reported as a diagnostic of where the difference comes from, never as the metric. |
| A velocity-likelihood term in the filter (v9 eq. 4-5) | The filter behind every result here does not have one; velocity is observed through prediction and the occupancy update over sweeps. |
| "20 Hz gives 0.785" | No run produces this; the source was never found. |
