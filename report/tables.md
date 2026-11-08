## Section 7 metrics

| set | images | found | crashed | corner err (%) | cell acc | empty F1 | exact | runtime (s) |
|---|---|---|---|---|---|---|---|---|
| dev | 20 | 20/20 | 0 | 0.063 | 0.9994 | 0.9995 | 19/20 | 2.13 |
| personal | 12 | 12/12 | 0 | 0.111 | 1.0000 | 1.0000 | 12/12 | 2.29 |

Cell accuracy, empty F1 and corner error are means over images where the puzzle was found. Empty F1 treats an empty cell as the positive class.

## By condition

| set | condition | images | found | crashed | corner err (%) | cell acc | empty F1 | exact | runtime (s) |
|---|---|---|---|---|---|---|---|---|---|
| dev | C1 | 5 | 5/5 | 0 | 0.050 | 1.0000 | 1.0000 | 5/5 | 2.11 |
| dev | C2 | 9 | 9/9 | 0 | 0.094 | 0.9986 | 0.9989 | 8/9 | 2.12 |
| dev | C3 | 4 | 4/4 | 0 | 0.045 | 1.0000 | 1.0000 | 4/4 | 2.18 |
| dev | C4 | 6 | 6/6 | 0 | 0.050 | 0.9979 | 0.9984 | 5/6 | 2.11 |
| dev | C5 | 6 | 6/6 | 0 | 0.049 | 1.0000 | 1.0000 | 6/6 | 2.15 |
| personal | C1 | 3 | 3/3 | 0 | 0.096 | 1.0000 | 1.0000 | 3/3 | 2.31 |
| personal | C2 | 4 | 4/4 | 0 | 0.131 | 1.0000 | 1.0000 | 4/4 | 2.26 |
| personal | C3 | 2 | 2/2 | 0 | 0.109 | 1.0000 | 1.0000 | 2/2 | 2.28 |
| personal | C4 | 2 | 2/2 | 0 | 0.106 | 1.0000 | 1.0000 | 2/2 | 2.32 |
| personal | C5 | 2 | 2/2 | 0 | 0.124 | 1.0000 | 1.0000 | 2/2 | 2.32 |

An image with several conditions is counted in each of its rows.

## OCR configurations

| set | mode | psm | cell acc | digit acc | exact | unread | empty F1 |
|---|---|---|---|---|---|---|---|
| dev | pipeline | 10+8 | 0.9994 | 0.9982 | 19/20 | 0 | 0.9995 |
| dev | single | 10 | 0.9802 | 0.9434 | 10/20 | 30 | 0.9855 |
| dev | single | 7 | 0.9802 | 0.9434 | 10/20 | 30 | 0.9855 |
| dev | single | 8 | 0.9932 | 0.9805 | 13/20 | 7 | 0.9962 |
| dev | single | 13 | 0.9932 | 0.9805 | 13/20 | 7 | 0.9962 |
| dev | single raw | 10 | 0.9235 | 0.7805 | 0/20 | 114 | 0.9483 |
| personal | pipeline | 10+8 | 1.0000 | 1.0000 | 12/12 | 0 | 1.0000 |
| personal | single | 10 | 0.9722 | 0.9206 | 1/12 | 27 | 0.9791 |
| personal | single | 7 | 0.9722 | 0.9206 | 1/12 | 27 | 0.9791 |
| personal | single | 8 | 0.9928 | 0.9794 | 6/12 | 5 | 0.9961 |
| personal | single | 13 | 0.9928 | 0.9794 | 6/12 | 5 | 0.9961 |
| personal | single raw | 10 | 0.8776 | 0.6500 | 1/12 | 106 | 0.9226 |

pipeline: psm 10 and 8 at 24, 32 and 40 px digit height (six reads); the most confident reading that passes the digit shape rules (pipeline/digit_rules.py) is kept. single: one call per filled cell, no fallback. raw: the margin-cropped gray cell without preparation. unread: filled cells for which OCR returned no digit (written as 0).

## Confusions (dev + personal, pipeline)

| true | predicted | count |
|---|---|---|
| 7 | 0 (empty) | 1 |

## Robustness: perturbed copies of all 32 annotated images

| perturbation | found | crashed | corner err mean (%) | corner err max (%) | cell acc | exact |
|---|---|---|---|---|---|---|
| original | 32/32 | 0 | 0.081 | 0.183 | 0.9996 | 31/32 |
| rot90 | 32/32 | 0 | 0.081 | 0.183 | 0.9996 | 31/32 |
| rot180 | 32/32 | 0 | 0.081 | 0.183 | 0.9996 | 31/32 |
| rot270 | 32/32 | 0 | 0.081 | 0.184 | 0.9996 | 31/32 |
| rot45 | 32/32 | 0 | 0.150 | 0.739 | 1.0000 | 32/32 |
| scale50 | 32/32 | 0 | 0.104 | 0.220 | 0.9992 | 30/32 |
| scale25 | 31/32 | 0 | 0.217 | 0.469 | 0.9968 | 28/32 |
| blur | 27/32 | 0 | 5.361 | 141.263 | 0.9634 | 14/32 |
| jpeg20 | 32/32 | 0 | 0.085 | 0.188 | 0.9985 | 28/32 |
| dark | 31/32 | 0 | 0.142 | 1.142 | 0.9996 | 30/32 |
| noise | 31/32 | 0 | 0.090 | 0.261 | 0.9980 | 27/32 |

rot90/180/270/45: in-plane rotation of the whole photo; scale50/25: downscaled; blur: Gaussian, sigma 2 px at 1200 px long side; jpeg20: JPEG quality 20; dark: gamma 2.5; noise: Gaussian, sigma 15 grey levels. Ground-truth corners are moved with the same transform.

## Harris parameters (8 images, one parameter at a time)

| parameter | value | candidates | not found | Harris peak err (%) | final err (%) | worst final (%) |
|---|---|---|---|---|---|---|
| HARRIS_BLOCK_RATIO | 0.003 | 454 | 0 | 0.401 | 0.059 | 0.125 |
| HARRIS_BLOCK_RATIO | 0.004 (chosen) | 378 | 0 | 0.508 | 0.059 | 0.124 |
| HARRIS_BLOCK_RATIO | 0.005 | 317 | 0 | 0.775 | 0.274 | 0.883 |
| HARRIS_BLOCK_RATIO | 0.007 | 242 | 0 | 0.984 | 0.856 | 1.003 |
| HARRIS_BLOCK_RATIO | 0.012 | 170 | 0 | 1.745 | 1.693 | 2.475 |
| HARRIS_KSIZE | 3 (chosen) | 378 | 0 | 0.508 | 0.059 | 0.124 |
| HARRIS_KSIZE | 5 | 348 | 0 | 0.513 | 0.059 | 0.124 |
| HARRIS_KSIZE | 7 | 322 | 0 | 0.507 | 0.059 | 0.124 |
| HARRIS_K | 0.02 | 387 | 0 | 0.505 | 0.059 | 0.124 |
| HARRIS_K | 0.04 (chosen) | 378 | 0 | 0.508 | 0.059 | 0.124 |
| HARRIS_K | 0.06 | 374 | 0 | 0.504 | 0.059 | 0.124 |
| HARRIS_K | 0.08 | 372 | 0 | 0.507 | 0.059 | 0.124 |
| HARRIS_K | 0.12 | 364 | 0 | 0.546 | 0.076 | 0.240 |
| THRESH_REL | 0.001 | 879 | 0 | 0.508 | 0.059 | 0.124 |
| THRESH_REL | 0.005 | 559 | 0 | 0.508 | 0.059 | 0.124 |
| THRESH_REL | 0.01 (chosen) | 378 | 0 | 0.508 | 0.059 | 0.124 |
| THRESH_REL | 0.05 | 208 | 0 | 0.508 | 0.059 | 0.124 |
| THRESH_REL | 0.1 | 166 | 0 | 0.508 | 0.059 | 0.124 |
| HARRIS_PEAK_REL | 0.01 | 378 | 0 | 0.508 | 0.059 | 0.124 |
| HARRIS_PEAK_REL | 0.05 | 378 | 0 | 0.508 | 0.059 | 0.124 |
| HARRIS_PEAK_REL | 0.1 (chosen) | 378 | 0 | 0.508 | 0.059 | 0.124 |
| HARRIS_PEAK_REL | 0.3 | 378 | 0 | 0.508 | 0.059 | 0.124 |
| HARRIS_PEAK_REL | 0.5 | 378 | 0 | 0.508 | 0.059 | 0.124 |
| HARRIS_SEARCH_PX | 3 | 378 | 0 | 0.508 | 0.059 | 0.124 |
| HARRIS_SEARCH_PX | 6 (chosen) | 378 | 0 | 0.508 | 0.059 | 0.124 |
| HARRIS_SEARCH_PX | 10 | 378 | 0 | 0.508 | 0.059 | 0.124 |
| HARRIS_SEARCH_PX | 15 | 378 | 0 | 0.508 | 0.059 | 0.124 |
| SUBPIX_WIN_PX | 0 | 378 | 0 | 0.508 | 0.508 | 0.759 |
| SUBPIX_WIN_PX | 2 | 378 | 0 | 0.508 | 0.175 | 0.282 |
| SUBPIX_WIN_PX | 3 (chosen) | 378 | 0 | 0.508 | 0.059 | 0.124 |
| SUBPIX_WIN_PX | 5 | 378 | 0 | 0.508 | 0.177 | 0.357 |
| SUBPIX_WIN_PX | 8 | 378 | 0 | 0.508 | 0.335 | 0.522 |
| SUBPIX_WIN_PX | 11 | 378 | 0 | 0.508 | 0.467 | 0.784 |

Images: dev_01, dev_07, dev_12, dev_15, dev_17, dev_20, s042401168_04, s042401168_08. Harris peak err is measured before cornerSubPix, final err after it.

## blockSize on all 32 annotated images

| blockSize ratio | mean err (%) | worst err (%) | images > 0.4 % |
|---|---|---|---|
| 0.003 | 0.093 | 0.557 | 1 |
| 0.004 | 0.081 | 0.183 | 0 |
| 0.005 | 0.345 | 0.955 | 11 |
| 0.007 | 0.841 | 1.526 | 30 |
| 0.012 | 1.594 | 2.475 | 32 |
