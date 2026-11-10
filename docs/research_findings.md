# Example Research Findings

This document shows how future measured results can be interpreted. Its values come
from the deterministic example package with seed 20261110.

## Main pattern

The example package gives Seq-LoRA a final ACC of 68.50% and FGT of 23.15
percentage points. Experience replay reaches 77.48% ACC and 10.40 points FGT.
Online EWC reaches 74.78% ACC and 12.17 points FGT. The joint reference reaches
83.36% ACC, but FGT does not apply because it is not trained sequentially.

The example paired analysis estimates an ER improvement of 15.63 points on AG News
and 10.83 points on SST-2. The corresponding example 95% bootstrap intervals are
12.96 to 18.38 points and 8.50 to 13.17 points. These values show the expected
reporting form. They are not claims about trained model performance.

## Resource interpretation

The example mean training time is 26.93 minutes for Seq-LoRA, 31.60 minutes for ER,
35.19 minutes for EWC, and 29.16 minutes for Joint. ER stores about 94 KB of replay
text. EWC stores about 8.7 MB of Fisher and anchor values.

The example replay sensitivity table shows a clear trade-off. Increasing the buffer
from 5% to 20% raises ACC from 73.6% to 79.1%, lowers FGT from 12.8 to 7.9 points,
and raises training time from 29.8 to 36.2 minutes.

## Hypothesis interpretation

Under these example values, H1 would be supported because both mitigation methods
show lower FGT and positive paired retention intervals. H2 would be supported
because replay improves retention while increasing training time and storage. A
measured study must evaluate the same rules against actual run files before making
either conclusion.

