# Data directory

Place the NASA C-MAPSS Turbofan Engine Degradation Simulation dataset here:

```
data/
├── train_FD001.txt   train_FD002.txt   train_FD003.txt   train_FD004.txt
├── test_FD001.txt    test_FD002.txt    test_FD003.txt    test_FD004.txt
└── RUL_FD001.txt      RUL_FD002.txt     RUL_FD003.txt     RUL_FD004.txt
```

Available from the NASA Prognostics Center of Excellence data repository, or
mirrored on Kaggle as "NASA Turbofan Jet Engine Data Set."

Each `train_*`/`test_*` file is whitespace-delimited with 26 columns: unit
number, cycle, 3 operating settings, and 21 sensor readings. `RUL_*` files
contain one integer per line — the true remaining useful life for the
corresponding engine in `test_*`.
