"""PSAS Launch-12 (TeleMetrum serial 1372, flight 2) flight-state case study for AeroDash.

Reference labels: onboard AltOS state_name. State codes/names, time, clock, voltages, RSSI/LQI and GPS
fields are never used as predictors. Outputs go to ./psas_results.
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SRC = Path(__file__).parent / "data" / "2015-07-19-serial-1372-flight-0002.csv"
OUT = Path(__file__).with_name("psas_results")
OUT.mkdir(exist_ok=True)

SEED = 42
SENTINEL = 2147483647
GAP = 20            # records excluded on each side of every test segment
N_FOLDS = 5
WINDOW_S = 1.0      # transition window (seconds) either side of a reference transition
MAIN_HEIGHT = 250.0  # "Main deploy" value from the CSV header

STATES = ["boost", "fast", "coast", "drogue", "main"]   # 'pad' (29 records) and 'landed' (1 record) are unsupported
SETS = {
    "A": ["acceleration", "pressure", "height", "speed", "temperature"],
    "B": ["acceleration", "pressure", "temperature"],
    "C": ["acceleration", "temperature"],
}
MEAS = ["acceleration", "pressure", "altitude", "height", "accel_speed", "baro_speed", "temperature"]


def load():
    t0 = time.perf_counter()
    hdr = [l for l in open(SRC) if l.startswith("#version")][0].lstrip("#").strip().split(",")
    seen, names = {}, []
    for c in hdr:
        seen[c] = seen.get(c, 0) + 1
        names.append(c if seen[c] == 1 else c + "_gps")
    raw = pd.read_csv(SRC, comment="#", header=None, names=names, skipinitialspace=True)
    info = {"raw_rows": len(raw)}
    df = raw[["time", "state", "state_name"] + MEAS].copy()
    info["sentinel_rows_in_predictors"] = int((df[MEAS] >= SENTINEL).any(axis=1).sum())
    df[MEAS] = df[MEAS].mask(df[MEAS] >= SENTINEL)
    df = df.dropna(subset=MEAS)
    info["after_sentinel_removal"] = len(df)
    # Repeated time + identical measurements: keep the last row (the post-transition state when states differ)
    dup = df.duplicated(subset=["time"] + MEAS, keep="last")
    info["exact_repeated_records_removed"] = int(dup.sum())
    info["repeated_with_state_change"] = int((df.loc[dup, "state"].values != df["state"].shift(-1)[dup].values).sum())
    df = df[~dup]
    info["distinct_timestamp_multi_record"] = int(df.time.duplicated().sum())
    info["speed_columns_identical"] = bool((df.accel_speed == df.baro_speed).all())
    info["altitude_minus_height_range"] = [float((df.altitude - df.height).min()), float((df.altitude - df.height).max())]
    df["speed"] = df["accel_speed"]
    info["landed_rows_before_exclusion"] = int((df.state_name == "landed").sum())
    info["pad_rows_before_exclusion"] = int((df.state_name == "pad").sum())
    df = df[df.state_name.isin(STATES)].reset_index(drop=True)
    info["retained_rows"] = len(df)
    info["time_start_s"], info["time_end_s"] = float(df.time.iloc[0]), float(df.time.iloc[-1])
    dt = df.time.diff().dropna()
    info["dt_median_s"], info["dt_max_s"] = float(dt.median()), float(dt.max())
    info["state_counts"] = df.state_name.value_counts().reindex(STATES).to_dict()
    info["state_intervals_s"] = {s: [float(df.time[df.state_name == s].min()), float(df.time[df.state_name == s].max())]
                                 for s in STATES}
    info["load_preprocess_s"] = time.perf_counter() - t0
    return df, info


def rule_baseline(X):
    """Stateless rule from the documented AltOS state definitions (fixed a priori, not fitted to labels).

    X columns: acceleration, height, speed.
    """
    a, h, v = X["acceleration"].values, X["height"].values, X["speed"].values
    out = np.empty(len(X), dtype=object)
    pad = (np.abs(v) < 10) & (np.abs(a) < 2) & (h < 10)
    asc = v > 0
    out[:] = np.where(h > MAIN_HEIGHT, "drogue", "main")
    out[asc & (a <= 0) & (v <= 200)] = "coast"
    out[asc & (a <= 0) & (v > 200)] = "fast"
    out[asc & (a > 0)] = "boost"
    out[pad] = "pad"
    return out


def make_models():
    return {
        "Gradient Boosting": lambda: make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=SEED)),
        "Random Forest": lambda: make_pipeline(StandardScaler(), RandomForestClassifier(random_state=SEED)),
        "Decision Tree": lambda: make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=SEED)),
        "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
        "kNN (k=5)": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Naive Bayes": lambda: make_pipeline(StandardScaler(), GaussianNB()),
    }


def state_blocked_folds(df, n_folds=N_FOLDS, gap=GAP):
    """Each state's contiguous interval is cut into n_folds contiguous segments; fold k tests segment k of every state.

    Records within `gap` rows of a test segment (either side) are removed from training.
    """
    segs = {k: [] for k in range(n_folds)}
    for s in STATES:
        idx = np.where(df.state_name.values == s)[0]
        for k, part in enumerate(np.array_split(idx, n_folds)):
            segs[k].append(part)
    folds = []
    for k in range(n_folds):
        test = np.concatenate(segs[k])
        excl = np.zeros(len(df), bool)
        for part in segs[k]:
            excl[max(0, part[0] - gap): part[-1] + gap + 1] = True
        train = np.where(~excl)[0]
        folds.append((train, test, segs[k]))
    return folds


def contiguous_folds(n, n_folds=N_FOLDS, gap=GAP):
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    out = []
    for k in range(n_folds):
        test = np.arange(edges[k], edges[k + 1])
        excl = np.zeros(n, bool)
        excl[max(0, test[0] - gap): test[-1] + gap + 1] = True
        out.append((np.where(~excl)[0], test))
    return out


def evaluate(df, folds, feats, model_names=None, collect_pred=True):
    y = df.state_name.values
    res, preds = {}, {}
    models = make_models()
    names = model_names or list(models)
    for name in ["Majority baseline", "Rule baseline"] + names:
        pred = np.empty(len(df), dtype=object)
        fold_f1, fit_t, inf_t = [], [], []
        for train, test, *_ in folds:
            Xtr, ytr, Xte = df.iloc[train][feats], y[train], df.iloc[test][feats]
            t0 = time.perf_counter()
            if name == "Majority baseline":
                maj = pd.Series(ytr).value_counts().idxmax()
                t1 = time.perf_counter()
                p = np.full(len(test), maj, dtype=object)
            elif name == "Rule baseline":
                t1 = time.perf_counter()
                p = rule_baseline(df.iloc[test])
            else:
                m = models[name]().fit(Xtr.values, ytr)
                t1 = time.perf_counter()
                p = m.predict(Xte.values)
            t2 = time.perf_counter()
            pred[test] = p
            fit_t.append(t1 - t0)
            inf_t.append((t2 - t1) / len(test) * 1000)  # ms per record
            fold_f1.append(f1_score(y[test], p, labels=STATES, average="macro", zero_division=0))
        mask = np.zeros(len(df), bool)
        for _, test, *_ in folds:
            mask[test] = True
        p_all, y_all = pred[mask].astype(str), y[mask]
        pr, rc, f1, sup = precision_recall_fscore_support(y_all, p_all, labels=STATES, zero_division=0)
        res[name] = {
            "accuracy": accuracy_score(y_all, p_all),
            "macro_f1": f1_score(y_all, p_all, labels=STATES, average="macro", zero_division=0),
            "macro_precision": float(pr.mean()), "macro_recall": float(rc.mean()),
            "fold_macro_f1_mean": float(np.mean(fold_f1)), "fold_macro_f1_sd": float(np.std(fold_f1, ddof=1)),
            "per_class": {s: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(sup[i])}
                          for i, s in enumerate(STATES)},
            "confusion": confusion_matrix(y_all, p_all, labels=STATES).tolist(),
            "fit_s_mean": float(np.mean(fit_t)), "infer_ms_per_record": float(np.mean(inf_t)),
        }
        preds[name] = pred
    return res, preds


def transition_windows(df, pred):
    y, t = df.state_name.values, df.time.values
    change = np.where(y[1:] != y[:-1])[0] + 1
    near = np.zeros(len(df), bool)
    events = []
    for i in change:
        w = np.abs(t - t[i]) <= WINDOW_S
        near |= w
        events.append({"from": y[i - 1], "to": y[i], "time_s": float(t[i]), "n_in_window": int(w.sum()),
                       "acc_in_window": float((pred[w] == y[w]).mean())})
    ok = pred == y
    return {"events": events, "n_events": len(events),
            "acc_transition": float(ok[near].mean()), "n_transition": int(near.sum()),
            "acc_other": float(ok[~near].mean()), "n_other": int((~near).sum())}


def plot_cms(panels, path):
    """panels: list of (title, confusion matrix) drawn side by side."""
    k = len(STATES)
    fig, axes = plt.subplots(1, len(panels), figsize=(10.5, 3.7))
    for ax, (title, cm) in zip(axes, panels):
        cm = np.array(cm)
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(k), STATES, rotation=45, fontsize=8); ax.set_yticks(range(k), STATES, fontsize=8)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
        ax.set_xlabel("Predicted", fontsize=8); ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Reference (onboard state)", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def plot_sync(df, preds, path):
    """preds: list of (label, predicted array). X axis is symlog so the 30 s ascent stays readable."""
    cmap = dict(zip(STATES, plt.cm.tab10.colors))
    n_bar = 1 + len(preds)
    fig, ax = plt.subplots(3 + n_bar, 1, figsize=(8, 7), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1.5, 1.5] + [.45] * n_bar, "hspace": .12})
    t = df.time.values
    ax[0].plot(t, df.height, lw=.8); ax[0].set_ylabel("Height (m)", fontsize=8)
    ax[1].plot(t, df.speed, lw=.8, color="C1"); ax[1].set_ylabel("Speed (m/s)", fontsize=8)
    ax[2].plot(t, df.acceleration, lw=.6, color="C2"); ax[2].set_ylabel("Accel (m/s²)", fontsize=8)
    for a, (lab, arr) in zip(ax[3:], [("Reference", df.state_name.values)] + preds):
        for s in STATES:
            m = arr == s
            a.scatter(t[m], np.zeros(m.sum()), c=[cmap[s]], marker="|", s=200, label=s if lab == "Reference" else None)
        if lab != "Reference":
            wrong = arr != df.state_name.values
            a.scatter(t[wrong], np.zeros(wrong.sum()), c="k", marker="x", s=6)
        a.set_yticks([]); a.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8)
    h, l = ax[3].get_legend_handles_labels()
    ax[0].legend(h, l, ncol=5, fontsize=7, loc="upper right", frameon=False)
    ax[-1].set_xscale("symlog", linthresh=40); ax[-1].set_xlim(-2, 800)
    ax[-1].set_xlabel("Time (s), symmetric-log axis", fontsize=8)
    fig.align_ylabels(ax)
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def clustering(df, feats):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    X = StandardScaler().fit_transform(df[feats].values)
    out = {"silhouette": {}, "ari": {}, "sizes": {}}
    labels = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        labels[k] = km.labels_
        out["silhouette"][k] = float(silhouette_score(X, km.labels_))
        out["ari"][k] = float(adjusted_rand_score(df.state_name.values, km.labels_))
        out["sizes"][k] = np.bincount(km.labels_).tolist()
    out["best_k"] = max(out["silhouette"], key=out["silhouette"].get)
    out["crosstab_k5"] = pd.crosstab(df.state_name, labels[5]).reindex(STATES).values.tolist()
    return out


def main():
    df, info = load()
    print(json.dumps(info, indent=1))
    y = df.state_name.values

    # ---- Primary: state-stratified blocked 5-fold ----
    folds = state_blocked_folds(df)
    fold_info = []
    for k, (tr, te, segs) in enumerate(folds):
        fold_info.append({"fold": k + 1, "n_train": len(tr), "n_test": len(te),
                          "train_classes": pd.Series(y[tr]).value_counts().reindex(STATES).fillna(0).astype(int).to_dict(),
                          "test_classes": pd.Series(y[te]).value_counts().reindex(STATES).fillna(0).astype(int).to_dict(),
                          "test_start_end_s": [[float(df.time[p[0]]), float(df.time[p[-1]])] for p in segs]})
    results = {"data": info, "folds": fold_info, "gap_records": GAP, "sets": {}}
    all_preds = {}
    for sname, feats in SETS.items():
        res, preds = evaluate(df, folds, feats)
        results["sets"][sname] = {"features": feats, "models": res}
        all_preds[sname] = preds
        print(f"\nSet {sname} {feats}")
        for n, r in res.items():
            print(f"  {n:20s} acc={r['accuracy']*100:5.1f} macroF1={r['macro_f1']:.3f} foldF1={r['fold_macro_f1_mean']:.3f}±{r['fold_macro_f1_sd']:.3f}")

    # ---- Contiguous (non-stratified) designs: class coverage ----
    n = len(df)
    cont = []
    for k, (tr, te) in enumerate(contiguous_folds(n)):
        cont.append({"fold": k + 1, "test_classes": sorted(set(y[te])), "train_classes": sorted(set(y[tr])),
                     "unseen_in_training": sorted(set(y[te]) - set(y[tr]))})
    results["contiguous_blocks"] = cont
    cut = int(n * 0.7)
    tr, te = np.arange(0, cut - GAP), np.arange(cut, n)
    chron = {"train_classes": sorted(set(y[tr])), "test_classes": sorted(set(y[te])),
             "unseen_in_training": sorted(set(y[te]) - set(y[tr])), "n_train": len(tr), "n_test": len(te),
             "test_class_counts": pd.Series(y[te]).value_counts().to_dict(), "models": {}}
    for name, mk in make_models().items():
        m = mk().fit(df.iloc[tr][SETS["A"]].values, y[tr]); p = m.predict(df.iloc[te][SETS["A"]].values)
        chron["models"][name] = {"accuracy": accuracy_score(y[te], p),
                                 "macro_f1_over_test_classes": f1_score(y[te], p, labels=sorted(set(y[te])), average="macro", zero_division=0)}
    results["chronological_final30"] = chron
    print("\nContiguous 5-block unseen classes:", [(c["fold"], c["unseen_in_training"]) for c in cont])
    print("Final-30% chronological:", chron["unseen_in_training"], chron["models"])

    # ---- Gap sensitivity (Set A) ----
    gs = {}
    for g in (0, 5, 20, 50, 100):
        r, _ = evaluate(df, state_blocked_folds(df, gap=g), SETS["A"], ["Gradient Boosting", "Random Forest", "Decision Tree"])
        gs[g] = {n: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"]} for n, v in r.items() if "baseline" not in n}
    results["gap_sensitivity_setA"] = gs
    print("\nGap sensitivity:", json.dumps(gs))

    # ---- Transition windows ----
    results["transition_windows"] = {}
    for sname in ("A", "B"):
        for name in ("Naive Bayes", "Random Forest", "Gradient Boosting", "Decision Tree", "Rule baseline"):
            results["transition_windows"][f"{sname}/{name}"] = transition_windows(df, all_preds[sname][name])
    for k, v in results["transition_windows"].items():
        print(k, round(v["acc_transition"], 3), v["n_transition"], round(v["acc_other"], 3), v["n_other"])

    # ---- Figures ----
    mA = results["sets"]["A"]["models"]
    plot_cms([("(a) Naive Bayes, Set A", mA["Naive Bayes"]["confusion"]),
              ("(b) Random Forest, Set A", mA["Random Forest"]["confusion"]),
              ("(c) Rule baseline", mA["Rule baseline"]["confusion"])], OUT / "cm_setA.png")
    plot_sync(df, [("Naive Bayes", all_preds["A"]["Naive Bayes"]), ("Rule", all_preds["A"]["Rule baseline"])], OUT / "sync_setA.png")
    results["clustering_setA"] = clustering(df, SETS["A"])
    cl = results["clustering_setA"]
    print("\nKMeans best k", cl["best_k"], {k: round(v, 3) for k, v in cl["silhouette"].items()}, {k: round(v, 3) for k, v in cl["ari"].items()})

    # ---- Software timing ----
    sw = {"load_preprocess_s_runs": []}
    for _ in range(5):
        _, i = load(); sw["load_preprocess_s_runs"].append(i["load_preprocess_s"])
    tr, te, _ = folds[0]
    sw["train_s"], sw["batch_infer_ms"], sw["single_record_ms"] = {}, {}, {}
    for name, mk in make_models().items():
        runs = []
        for _ in range(3):
            t0 = time.perf_counter(); m = mk().fit(df.iloc[tr][SETS["A"]].values, y[tr]); runs.append(time.perf_counter() - t0)
        sw["train_s"][name] = runs
        Xte = df.iloc[te][SETS["A"]].values
        b = {}
        for bs in (1, 100, len(Xte)):
            r_ = []
            for _ in range(5):
                t0 = time.perf_counter(); m.predict(Xte[:bs]); r_.append((time.perf_counter() - t0) * 1000)
            b[bs] = r_
        sw["batch_infer_ms"][name] = b
        # paced-replay style loop: one predict call per record, as in a live feed
        lat = []
        for row in Xte[:300]:
            t0 = time.perf_counter(); m.predict(row.reshape(1, -1)); lat.append((time.perf_counter() - t0) * 1000)
        sw["single_record_ms"][name] = {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95)),
                                        "max": float(np.max(lat)), "max_rate_hz": float(1000 / np.mean(lat))}
    sw["n_train_fold1"], sw["n_test_fold1"] = len(tr), len(te)
    import platform, sklearn
    sw["env"] = {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__,
                 "numpy": np.__version__, "platform": platform.platform(), "processor": platform.processor()}
    first = np.where(y[1:] != y[:-1])[0] + 1
    results["transition_rows"] = [{"to": y[i], "time_s": float(df.time[i]), "acceleration": float(df.acceleration[i]),
                                   "height": float(df.height[i]), "speed": float(df.speed[i])} for i in first]
    results["software"] = sw
    print("\nload/preprocess s:", np.round(sw["load_preprocess_s_runs"], 3))
    for n_ in sw["single_record_ms"]:
        print(n_, "train", round(np.mean(sw["train_s"][n_]), 3), "s; single-record", sw["single_record_ms"][n_])

    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=1, default=str)
    rows = []
    for sname, s in results["sets"].items():
        for n_, r in s["models"].items():
            rows.append({"set": sname, "model": n_, "accuracy_pct": round(r["accuracy"] * 100, 1), "macro_f1": round(r["macro_f1"], 3),
                         "fold_macro_f1_mean": round(r["fold_macro_f1_mean"], 3), "fold_macro_f1_sd": round(r["fold_macro_f1_sd"], 3),
                         "infer_ms_per_record": round(r["infer_ms_per_record"], 4), **{f"f1_{k}": round(v["f1"], 3) for k, v in r["per_class"].items()}})
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)


if __name__ == "__main__":
    main()
