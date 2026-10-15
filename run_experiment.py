"""Reproduce exploratory ORBIAGE baseline and uncertainty ranking experiment."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def norm_rank(x):
    return rankdata(x, method="average") / len(x)

def run(data_path, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    required = {"train","anomaly","segment","channel"}
    if not required.issubset(df.columns):
        raise ValueError("Missing required columns: " + str(required-set(df.columns)))
    train = df.loc[df.train==1].copy()
    test = df.loc[df.train==0].copy()
    y, yt = train.anomaly.to_numpy(), test.anomaly.to_numpy()
    cols = [c for c in df.columns if c not in ("anomaly","train","segment")]
    num = [c for c in cols if c != "channel"]
    pre = ColumnTransformer([
        ("num", Pipeline([("impute",SimpleImputer(strategy="median")),
                         ("scale",StandardScaler())]), num),
        ("channel", OneHotEncoder(handle_unknown="ignore",sparse_output=False), ["channel"])
    ])
    A = pre.fit_transform(train[cols])
    B = pre.transform(test[cols])
    train_idx, val_idx = train_test_split(np.arange(len(y)), test_size=.25,
                                         random_state=42, stratify=y)
    results = []
    def calc(name, p, probability=True):
        row = {"method":name,"ROC AUC":roc_auc_score(yt,p),
               "PR AUC":average_precision_score(yt,p),
               "Brier":brier_score_loss(yt,np.clip(p,0,1)) if probability else np.nan}
        for k in [10,20,30,50,75,100]:
            top = np.argsort(-p,kind="stable")[:k]
            row[f"TP@{k}"] = int(yt[top].sum())
            row[f"Precision@{k}"] = float(yt[top].mean())
        results.append(row)
    lr = LogisticRegression(max_iter=3000,class_weight="balanced").fit(A,y)
    calc("Logistic regression",lr.predict_proba(B)[:,1])
    opts = dict(n_estimators=350,min_samples_leaf=3,random_state=42,
                n_jobs=-1,class_weight="balanced_subsample")
    rf = RandomForestClassifier(**opts).fit(A,y)
    calc("Random forest",rf.predict_proba(B)[:,1])
    calibrated = CalibratedClassifierCV(RandomForestClassifier(**opts),method="sigmoid",cv=5)
    calibrated.fit(A,y)
    p = calibrated.predict_proba(B)[:,1]
    calc("RF + sigmoid calibration",p)
    rng = np.random.default_rng(43)
    ensemble = []
    for i in range(15):
        ix = rng.choice(len(y),len(y),replace=True)
        m = RandomForestClassifier(n_estimators=90,min_samples_leaf=3,
                random_state=200+i,n_jobs=-1,class_weight="balanced_subsample")
        m.fit(A[ix],y[ix])
        ensemble.append(m.predict_proba(B)[:,1])
    unc = np.std(ensemble,axis=0)
    neighbors = NearestNeighbors(n_neighbors=10).fit(A)
    distance = neighbors.kneighbors(B)[0].mean(axis=1)
    m = RandomForestClassifier(**opts).fit(A[train_idx],y[train_idx])
    pv = m.predict_proba(A[val_idx])[:,1]
    ens_val = []
    for i in range(8):
        ix = rng.choice(train_idx,len(train_idx),replace=True)
        mm = RandomForestClassifier(n_estimators=60,min_samples_leaf=3,random_state=400+i,n_jobs=-1)
        mm.fit(A[ix],y[ix])
        ens_val.append(mm.predict_proba(A[val_idx])[:,1])
    uv = np.std(ens_val,axis=0)
    weights = [0,.05,.10,.20,.35,.50]
    w = max(weights,key=lambda t: average_precision_score(
        y[val_idx],(1-t)*norm_rank(pv)+t*norm_rank(uv)))
    calc(f"ORBIAGE uncertainty blend (w={w})",
         (1-w)*norm_rank(p)+w*norm_rank(unc), probability=False)
    dv = neighbors.kneighbors(A[val_idx])[0].mean(axis=1)
    shift_weights = [0,.05,.10,.20,.35]
    z = max(shift_weights,key=lambda t: average_precision_score(
        y[val_idx],(1-t)*norm_rank(pv)+t*norm_rank(dv)))
    calc(f"ORBIAGE shift blend (w={z})",
         (1-z)*norm_rank(p)+z*norm_rank(distance), probability=False)
    for fraction in [.1,.2]:
        calc(f"ORBIAGE exploratory (uncertainty +{fraction})",
             norm_rank(p)+fraction*norm_rank(unc),probability=False)
    results = pd.DataFrame(results)
    results.to_csv(output/"real_results.csv",index=False)
    print(f"train={len(train)}, test={len(test)}, train anomalies={sum(y)}, test anomalies={sum(yt)}")
    print(results[["method","ROC AUC","PR AUC","TP@100"]].to_string(index=False))
    fig,ax = plt.subplots(figsize=(9,5))
    ks = [10,20,30,50,75,100]
    for label in ["Random forest","RF + sigmoid calibration",
                  f"ORBIAGE uncertainty blend (w={w})",f"ORBIAGE shift blend (w={z})"]:
        row = results[results.method==label].iloc[0]
        ax.plot(ks,[row[f"TP@{k}"] for k in ks],marker="o",label=label)
    ax.set(xlabel="Number of alerts reviewed",ylabel="True anomalies recovered",
           title="OPS-SAT: segment-level alert triage")
    ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output/"real_alert_budget.png",dpi=190)
    plt.close(fig)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",default="data/dataset.csv")
    parser.add_argument("--output",default="results")
    args=parser.parse_args()
    run(args.data,args.output)
