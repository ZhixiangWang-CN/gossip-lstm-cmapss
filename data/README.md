# NASA C-MAPSS FD001

`PM_train.txt`, `PM_test.txt` and `PM_truth.txt` are the FD001 subset (single operating condition, single fault mode) of the
NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) turbofan degradation dataset, redistributed here
unchanged in content for reproducibility. The data are provided by the NASA Prognostics Center of Excellence:
https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

Columns: engine id, cycle, 3 operational settings, 21 sensor measurements. `PM_truth.txt` gives the true remaining useful
life at the last observed cycle of each test engine.

Please cite: A. Saxena, K. Goebel, D. Simon and N. Eklund, "Damage propagation modeling for aircraft engine run-to-failure
simulation," *2008 International Conference on Prognostics and Health Management*, pp. 1–9.
https://doi.org/10.1109/PHM.2008.4711414
