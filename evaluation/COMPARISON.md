# Experiment Comparison

| Measure | 3-epoch baseline | Expanded 24-epoch run | Change |
|---|---:|---:|---:|
| Total generated records | 3,250 | 10,000 | +6,750 |
| Training records | 2,400 | 7,500 | +5,100 |
| Validation records | 300 | 1,000 | +700 |
| Test records | 400 | 1,000 | +600 |
| Challenge records | 150 | 500 | +350 |
| Test false clearances | 0 | 0 | 0 |
| Test attention recall | 100% | 100% | 0 points |
| Test clearance coverage | 28.25% | 31.50% | +3.25 points |
| Test pair consistency | 56.50% | 63.00% | +6.50 points |
| Challenge false clearances | 0 | 0 | 0 |
| Challenge attention recall | 100% | 100% | 0 points |
| Challenge clearance coverage | 7.33% | 12.60% | +5.27 points |
| Challenge pair consistency | 14.67% | 25.20% | +10.53 points |

The expanded run changed both the generated data and the number of epochs. The table does not
attribute the gains to one factor. The larger dataset introduced indirect trigger descriptions,
split-specific formats, and explicit negations of unrelated risk conditions.

Both experiments selected a threshold with zero false clearances on their own validation splits.
The same result held on their generated test and challenge splits. This is an experimental
observation, not a claim about real legal requests or other datasets.
