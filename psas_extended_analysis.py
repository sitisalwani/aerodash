"""Extended analyses for the PSAS Launch-12 case study (complements psas_analysis.py).

Reuses the loading, fold design and model definitions of psas_analysis.py so every number is
directly comparable with Tables VII-IX of the paper. Outputs go to ./psas_results/extended.

E1  Gap justification: autocorrelation of predictors (raw and first-differenced), in records and seconds.
E2  Temperature as a clock proxy: temperature-time correlation, per-state temperature ranges,
    and ablations with temperature removed / temperature alone / acceleration alone.
E3  Fold extrapolation diagnostic: per-fold macro-F1 and how far each test segment lies outside
    the training range of its own class (height).
E4  Stateful (ordered) rule baseline: the same documented thresholds as the stateless rule,
    but states may only advance boost -> fast -> coast -> drogue -> main, as in AltOS.
E5  Sequence-aware ML: left-to-right HMM decoding of out-of-fold class probabilities
    (causal forward filter = usable online; Viterbi = offline/post-flight).
E6  Causal temporal features (trailing 0.5 s windows) for the stateless classifiers.
E7  Uncertainty: moving-block bootstrap 95% CIs for macro-F1 and paired differences vs the rule.
E8  Sensitivity of the rule baseline to its data-informed pad tolerances and the 200 m/s threshold.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import psas_analysis as P

OUT = P.OUT / "extended"
OUT.mkdir(parents=True, exist_ok=True)
STATES = P.STATES
IDX = {s: i for i, s in enumerate(STATES)}
RNG = np.random.default_rng(P.SEED)


def scores(y, p):
    return {"accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, labels=STATES, average="macro", zero_division=0))}


# ---------------------------------------------------------------- E1
def autocorr_table(df):
    rows = []
    for c in ["acceleration", "pressure", "height", "speed", "temperature"]:
        for kind, x in (("raw", df[c]), ("first difference", df[c].diff())):
            rows.append({"feature": c, "series": kind, **{f"lag{l}": round(float(x.autocorr(l)), 3) for l in (1, 20, 100, 500)}})
    return rows


# ---------------------------------------------------------------- E3
def fold_diagnostic(df, folds, preds):
    y, h = df.state_name.values, df.height.values
    out = []
    for k, (tr, te, segs) in enumerate(folds):
        row = {"fold": k + 1}
        for name in ("Rule baseline", "Naive Bayes", "Random Forest"):
            row[f"macroF1_{name}"] = round(f1_score(y[te], preds[name][te].astype(str), labels=STATES, average="macro", zero_division=0), 3)
        # share of each class's test records whose height is outside that class's training height range
        for s, seg in zip(STATES, segs):
            htr = h[tr][y[tr] == s]
            row[f"{s}_height_test"] = f"{h[seg].min():.0f}..{h[seg].max():.0f}"
            row[f"{s}_outside_train_range_pct"] = round(100 * float(((h[seg] < htr.min()) | (h[seg] > htr.max())).mean()), 1)
        out.append(row)
    return out


# ---------------------------------------------------------------- E4
def stateful_rule(df, main_h=P.MAIN_HEIGHT, fast_v=200.0):
    """Ordered version of rule_baseline: identical thresholds, but the state can only advance."""
    a, v, h = df.acceleration.values, df.speed.values, df.height.values
    s, out = 0, np.empty(len(df), dtype=object)
    for i in range(len(df)):
        if s == 0 and a[i] <= 0:                          # burnout
            s = 1 if v[i] > fast_v else 2
        if s == 1 and v[i] <= fast_v:                     # below fast threshold
            s = 2
        if s == 2 and v[i] <= 0:                          # apogee
            s = 3
        if s == 3 and h[i] <= main_h:                     # main deploy altitude
            s = 4
        out[i] = STATES[s]
    return out


# ---------------------------------------------------------------- E5
def oof_proba(df, folds, feats, name):
    """Out-of-fold class probabilities in STATES order (each record scored by a model that never saw it)."""
    y = df.state_name.values
    proba = np.zeros((len(df), len(STATES)))
    for tr, te, *_ in folds:
        m = P.make_models()[name]().fit(df.iloc[tr][feats].values, y[tr])
        p = m.predict_proba(df.iloc[te][feats].values)
        for j, c in enumerate(m.classes_):
            proba[te, IDX[c]] = p[:, j]
    return proba


def left_to_right(n_states, eps=1e-3):
    T = np.eye(n_states) * (1 - eps)
    for i in range(n_states - 1):
        T[i, i + 1] = eps
    T[-1, -1] = 1.0
    return np.log(np.clip(T, 1e-300, None))


def emissions(proba, prior):
    """Hybrid HMM: scaled likelihood p(x|s) ~ p(s|x) / p(s)."""
    return np.log(np.clip(proba, 1e-9, None)) - np.log(prior)


def viterbi(logE, logT):
    n, k = logE.shape
    d = np.full(k, -np.inf); d[0] = 0.0
    d = d + logE[0]
    bp = np.zeros((n, k), int)
    for t in range(1, n):
        cand = d[:, None] + logT
        bp[t] = cand.argmax(0)
        d = cand.max(0) + logE[t]
    path = np.empty(n, int); path[-1] = d.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return np.array(STATES, dtype=object)[path]


def forward_filter(logE, logT):
    """Causal: decision at t uses only records <= t (usable in a live dashboard)."""
    n, k = logE.shape
    a = np.full(k, -np.inf); a[0] = 0.0
    a = a + logE[0]; a -= np.logaddexp.reduce(a)
    out = np.empty(n, int); out[0] = a.argmax()
    for t in range(1, n):
        a = np.logaddexp.reduce(a[:, None] + logT, axis=0) + logE[t]
        a -= np.logaddexp.reduce(a)
        out[t] = a.argmax()
    return np.array(STATES, dtype=object)[out]


# ---------------------------------------------------------------- E6
def add_causal_features(df, win_s=0.5):
    d = df.copy()
    td = pd.to_timedelta(d.time, unit="s")
    r = d.set_index(td)
    d["acc_mean_0.5s"] = r.acceleration.rolling(f"{int(win_s*1000)}ms").mean().values
    d["acc_std_0.5s"] = r.acceleration.rolling(f"{int(win_s*1000)}ms").std().fillna(0).values
    d["dh_0.5s"] = (r.height - r.height.rolling(f"{int(win_s*1000)}ms").apply(lambda x: x[0], raw=True)).values
    return d


# ---------------------------------------------------------------- E7
def block_bootstrap(y, preds, ref="Rule baseline", block=100, B=1000):
    n = len(y)
    starts = np.arange(0, n - block + 1)
    nb = int(np.ceil(n / block))
    names = list(preds)
    samples = {k: [] for k in names}
    diffs = {k: [] for k in names if k != ref}
    for _ in range(B):
        idx = np.concatenate([np.arange(s, s + block) for s in RNG.choice(starts, nb)])[:n]
        f = {k: f1_score(y[idx], preds[k][idx], labels=STATES, average="macro", zero_division=0) for k in names}
        for k in names:
            samples[k].append(f[k])
        for k in diffs:
            diffs[k].append(f[k] - f[ref])
    ci = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
    return {"block_records": block, "B": B,
            "macro_f1_ci95": {k: ci(v) for k, v in samples.items()},
            f"diff_vs_{ref}_ci95": {k: ci(v) for k, v in diffs.items()},
            f"P(model >= {ref})": {k: round(float(np.mean(np.array(v) >= 0)), 3) for k, v in diffs.items()}}


# ---------------------------------------------------------------- E8
def rule_sensitivity(df):
    y = df.state_name.values
    rows = []
    orig = P.MAIN_HEIGHT
    for vs, as_, hs in [(10, 2, 10), (5, 1, 5), (20, 4, 20), (0, 0, 0)]:
        for fast_v in (150, 200, 250):
            a, h, v = df.acceleration.values, df.height.values, df.speed.values
            out = np.empty(len(df), dtype=object)
            asc = v > 0
            out[:] = np.where(h > orig, "drogue", "main")
            out[asc & (a <= 0) & (v <= fast_v)] = "coast"
            out[asc & (a <= 0) & (v > fast_v)] = "fast"
            out[asc & (a > 0)] = "boost"
            out[(np.abs(v) < vs) & (np.abs(a) < as_) & (h < hs)] = "pad"
            rows.append({"pad_tol(|v|,|a|,h)": f"{vs},{as_},{hs}", "fast_v": fast_v, **scores(y, out.astype(str))})
    return rows


def plot_decoding(df, series, path):
    cmap = dict(zip(STATES, plt.cm.tab10.colors))
    t, yref = df.time.values, df.state_name.values
    fig, ax = plt.subplots(len(series) + 1, 1, figsize=(8, 0.55 * (len(series) + 1) + 0.6), sharex=True)
    for a, (lab, arr) in zip(ax, [("Reference", yref)] + series):
        for s in STATES:
            m = arr == s
            a.scatter(t[m], np.zeros(m.sum()), c=[cmap[s]], marker="|", s=200, label=s if lab == "Reference" else None)
        if lab != "Reference":
            w = arr != yref
            a.scatter(t[w], np.zeros(w.sum()), c="k", marker="x", s=6)
        a.set_yticks([]); a.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8)
    ax[0].legend(ncol=5, fontsize=7, loc="lower right", bbox_to_anchor=(1, 1), frameon=False)
    ax[-1].set_xscale("symlog", linthresh=40); ax[-1].set_xlim(-2, 800)
    ax[-1].set_xlabel("Time (s), symmetric-log axis", fontsize=8)
    fig.align_ylabels(ax); fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def main():
    df, _ = P.load()
    y = df.state_name.values
    folds = P.state_blocked_folds(df)
    R = {}

    # E1
    R["E1_autocorrelation"] = autocorr_table(df)
    R["E1_sampling_hz"] = {s: round(float((df.state_name == s).sum() / np.ptp(df.time[df.state_name == s])), 1) for s in STATES}

    # E2
    R["E2_temperature"] = {"corr_temperature_time": round(float(df.temperature.corr(df.time)), 3),
                           "per_state_range_C": {s: [float(df.temperature[df.state_name == s].min()),
                                                     float(df.temperature[df.state_name == s].max())] for s in STATES}}
    extra_sets = {"A-noT": ["acceleration", "pressure", "height", "speed"],
                  "T": ["temperature"], "Acc": ["acceleration"],
                  "Phys": ["acceleration", "height", "speed"]}
    R["E2_ablation"] = {}
    for sname, feats in extra_sets.items():
        res, _ = P.evaluate(df, folds, feats)
        R["E2_ablation"][sname] = {"features": feats, **{n: {"accuracy": round(r["accuracy"], 3), "macro_f1": round(r["macro_f1"], 3)}
                                                         for n, r in res.items() if n != "Rule baseline"}}

    # baseline predictions for Set A (re-run so this script is self-contained)
    resA, predsA = P.evaluate(df, folds, P.SETS["A"])
    R["E3_fold_diagnostic"] = fold_diagnostic(df, folds, predsA)

    # E4
    sr = stateful_rule(df)
    R["E4_stateful_rule"] = {**scores(y, sr), "transition_windows": P.transition_windows(df, sr)}

    # E5
    prior = np.bincount([IDX[s] for s in y], minlength=len(STATES)) / len(y)
    logT = left_to_right(len(STATES))
    R["E5_sequence_decoding"] = {}
    decoded = {}
    for name in ("Naive Bayes", "Random Forest", "Gradient Boosting", "Logistic Regression"):
        pr = oof_proba(df, folds, P.SETS["A"], name)
        logE = emissions(pr, prior)
        fw, vt = forward_filter(logE, logT), viterbi(logE, logT)
        decoded[f"{name} + forward"], decoded[f"{name} + Viterbi"] = fw, vt
        R["E5_sequence_decoding"][name] = {
            "stateless": scores(y, predsA[name].astype(str)),
            "causal_forward_filter": {**scores(y, fw), "acc_transition": round(P.transition_windows(df, fw)["acc_transition"], 3)},
            "offline_viterbi": {**scores(y, vt), "acc_transition": round(P.transition_windows(df, vt)["acc_transition"], 3)},
            "viterbi_transition_times_s": [float(df.time[i]) for i in np.where(vt[1:] != vt[:-1])[0] + 1],
        }
    R["reference_transition_times_s"] = [float(df.time[i]) for i in np.where(y[1:] != y[:-1])[0] + 1]

    # E6
    dfc = add_causal_features(df)
    feats_t = P.SETS["A"] + ["acc_mean_0.5s", "acc_std_0.5s", "dh_0.5s"]
    feats_tnoT = [f for f in feats_t if f != "temperature"]
    R["E6_causal_features"] = {}
    for lab, fs in (("A+temporal", feats_t), ("A+temporal-noT", feats_tnoT)):
        res, _ = P.evaluate(dfc, folds, fs, ["Naive Bayes", "Random Forest", "Gradient Boosting", "Logistic Regression"])
        R["E6_causal_features"][lab] = {n: {"accuracy": round(r["accuracy"], 3), "macro_f1": round(r["macro_f1"], 3)}
                                        for n, r in res.items() if "baseline" not in n}

    # E7
    boot_preds = {"Rule baseline": predsA["Rule baseline"].astype(str), "Stateful rule": sr.astype(str),
                  "Naive Bayes": predsA["Naive Bayes"].astype(str), "Random Forest": predsA["Random Forest"].astype(str),
                  "Naive Bayes + forward": decoded["Naive Bayes + forward"].astype(str),
                  "Naive Bayes + Viterbi": decoded["Naive Bayes + Viterbi"].astype(str)}
    R["E7_bootstrap"] = block_bootstrap(y, boot_preds)

    # E8
    R["E8_rule_sensitivity"] = rule_sensitivity(df)

    plot_decoding(df, [("Rule (stateless)", predsA["Rule baseline"]), ("Rule (stateful)", sr),
                       ("NB (stateless)", predsA["Naive Bayes"]), ("NB + forward", decoded["Naive Bayes + forward"]),
                       ("NB + Viterbi", decoded["Naive Bayes + Viterbi"])], OUT / "sequence_decoding.png")

    with open(OUT / "extended_results.json", "w") as f:
        json.dump(R, f, indent=1, default=str)

    # concise console report
    pd.set_option("display.width", 200)
    print("E1 sampling Hz", R["E1_sampling_hz"]); print(pd.DataFrame(R["E1_autocorrelation"]).to_string(index=False))
    print("\nE2", R["E2_temperature"])
    for k, v in R["E2_ablation"].items():
        print(k, v["features"], {n: (x["accuracy"], x["macro_f1"]) for n, x in v.items() if n != "features"})
    print("\nE3"); print(pd.DataFrame(R["E3_fold_diagnostic"]).T.to_string())
    print("\nE4 stateful rule", {k: v for k, v in R["E4_stateful_rule"].items() if k != "transition_windows"},
          "acc_transition", round(R["E4_stateful_rule"]["transition_windows"]["acc_transition"], 3))
    print("\nE5")
    for k, v in R["E5_sequence_decoding"].items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "viterbi_transition_times_s"})
        print("   viterbi transitions", v["viterbi_transition_times_s"], "ref", R["reference_transition_times_s"])
    print("\nE6", json.dumps(R["E6_causal_features"]))
    print("\nE7", json.dumps(R["E7_bootstrap"], indent=1))
    print("\nE8"); print(pd.DataFrame(R["E8_rule_sensitivity"]).to_string(index=False))


if __name__ == "__main__":
    main()
