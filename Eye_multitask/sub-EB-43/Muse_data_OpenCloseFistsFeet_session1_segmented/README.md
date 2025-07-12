Muse_data_OpenCloseFeet_segmented/
│
├── exp_1/                       # Experiment 1 → open / close fists
│   ├── openclosefists_run1_AF7.csv
│   ├── openclosefists_run1_AF8.csv
│   ├── openclosefists_run1_TP9.csv
│   ├── openclosefists_run1_TP10.csv
│   ├── run_1_label.csv
│   ├── openclosefists_run2_*.csv
│   └── run_2_label.csv
│
└── exp_2/                       # Experiment 2 → open / close feet
    ├── openclosefeet_run1_AF7.csv
    ├── openclosefeet_run1_AF8.csv
    ├── openclosefeet_run1_TP9.csv
    ├── openclosefeet_run1_TP10.csv
    ├── run_1_label.csv
    ├── openclosefeet_run2_*.csv
    └── run_2_label.csv


Note that each experiment contains 2 runs. exp_1 first run has only 63 samples due to instable connect to the EEG.

Regarding labels:
0 -> non-target (open fist, open feet)
1 -> target (close fist, close feet)
