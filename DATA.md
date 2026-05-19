# Data layout

Do not commit datasets to this repository. Download the datasets separately and place them under `data/`.

Expected layout:

```text
data/
  darcy/
    train/p_data.csv
    train/K_data.csv
    valid/p_data.csv
    valid/K_data.csv
  mechanics/
    train/fields/*.npy
    test/valid/fields/*.npy
    test/test_level_1/fields/*.npy
    test/test_level_2/fields/*.npy
    solidspy_k_no_BC/...
  ch_2Dxysec.pickle        # turbulent channel-flow pickle
```

The charge/Poisson task is generated synthetically by `src_pr.data_utils.DatasetCharge`, so it does not require a downloaded dataset.
