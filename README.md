# AeroDash: flight-state classification analysis

Analysis code for the AeroDash rocket-telemetry study. It evaluates six supervised classifiers and K-Means clustering against majority-class and rule-based baselines. It uses two datasets: a recording made with the AeroDash prototype (LoRa / ESP32-C3), and the public PSAS Launch-12 flight log (AltOS TeleMetrum v2.0, serial 1372, flight 2).

## Repository layout

| Path | Contents |
|---|---|
| `psas_analysis.py` | Main PSAS case study. Loads and cleans the data, builds state-stratified blocked 5-fold CV, runs the classifiers, baselines, feature ablation (Sets A–C), gap sensitivity, transition-window accuracy, clustering, timing, and the figures. |
| `psas_extended_analysis.py` | Extended diagnostics: autocorrelation, temperature-as-clock ablation, per-fold extrapolation, a stateful (ordered) rule, left-to-right HMM decoding (forward filter and Viterbi), causal temporal features, block-bootstrap CIs, and rule sensitivity. |
| `psas_results/` | Outputs of `psas_analysis.py` (`results.json`, `summary.csv`, figures). |
| `psas_results/extended/` | Outputs of `psas_extended_analysis.py`. |
| `data/aerodash_recording.csv` | AeroDash prototype recording (1,545 records, ~139 s, mostly stationary). |

## Data

**AeroDash recording** is included in `data/aerodash_recording.csv`.

**PSAS Launch-12** is not redistributed here. Download `2015-07-19-serial-1372-flight-0002.csv` from the Portland State Aerospace Society repository at commit `abca095430eda435ac3e9bebebba2ade6e5943f9`:

https://github.com/psas/Launch-12/tree/abca095430eda435ac3e9bebebba2ade6e5943f9

Place it at `data/2015-07-19-serial-1372-flight-0002.csv`.

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python psas_analysis.py            # ~1-2 min, writes psas_results/
python psas_extended_analysis.py   # ~4-5 min, writes psas_results/extended/
```

Results in the paper were produced with Python 3.13, scikit-learn 1.9.1, pandas 3.1.0 and NumPy 2.5.4. All models use library defaults with seed 42. Timing numbers depend on the machine.

## Notes

- Labels for PSAS are the onboard AltOS `state_name`. State fields, time, clock, voltages, RSSI/LQI and GPS columns are never used as predictors.
- Scores are *retrospective within-flight* results: training folds include records recorded after the test segments. They do not measure cross-flight generalisation.
- The AeroDash classification script (chronological hold-out and blocked CV tables) will be added separately.

## License

Code is released under the MIT License (see `LICENSE`). The PSAS data remain subject to their owners' terms.
