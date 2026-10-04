import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except ImportError:
    display = None

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, silhouette_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

"""\nStandalone consumer consumption analysis and prediction project.\n\nThis file combines the original data utilities, CBLOF anomaly detection,\nrooftop-solar scenario calculation, and ML training pipeline into one file.\n"""

# ========================= DATA UTILITIES =========================

MONTH_START = 7

def load_data(path):
    df = pd.read_excel(path)
    month_cols = list(df.columns[MONTH_START:])
    for c in month_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, month_cols

def build_features(df, month_cols):
    out = df.copy()
    first7, last3 = month_cols[:7], month_cols[7:]
    out["train_avg"] = out[first7].mean(axis=1)
    out["train_std"] = out[first7].std(axis=1)
    out["train_min"] = out[first7].min(axis=1)
    out["train_max"] = out[first7].max(axis=1)
    out["train_total"] = out[first7].sum(axis=1, min_count=1)
    out["future_avg"] = out[last3].mean(axis=1)
    out["future_total"] = out[last3].sum(axis=1, min_count=1)
    out["cv"] = out["train_std"] / out["train_avg"].replace(0, np.nan)
    out["consumption_per_kw"] = out["train_avg"] / out["CONTRACT_LOAD"].replace(0, np.nan)
    return out, first7, last3

def make_target(df):
    threshold = df["train_avg"].quantile(0.75)
    out = df.copy()
    out["high_consumption_target"] = (out["future_avg"] >= threshold).astype(int)
    return out, threshold

# ========================= CBLOF ANOMALY DETECTION =========================

"""Clustering-based anomaly scores used in the project review queue.

The implementation follows the CBLOF idea: K-Means forms behavioral groups,
cluster sizes define large and small groups, and distance to a large group
contributes to the review score. Keeping the calculation here makes the
assumptions visible and avoids an extra runtime dependency.
"""

def cblof_scores(X, n_clusters=4, alpha=0.9, beta=0.2, random_state=42):
    X=np.asarray(X,dtype=float)
    km=KMeans(n_clusters=n_clusters,n_init=20,random_state=random_state)
    labels=km.fit_predict(X)
    centers=km.cluster_centers_
    sizes=np.bincount(labels,minlength=n_clusters)
    order=np.argsort(sizes)[::-1]
    cumulative=np.cumsum(sizes[order])
    large=set(order[cumulative <= alpha*len(X)])
    if not large:
        large={order[0]}
    large_sizes=sizes[list(large)]
    beta_count=max(1,int(np.max(large_sizes)*beta))
    large=set([i for i in large if sizes[i]>=beta_count]) or {order[0]}

    d=np.linalg.norm(X[:,None,:]-centers[None,:,:],axis=2)
    nearest_large=np.min(d[:,list(large)],axis=1)
    assigned_size=sizes[labels]
    score=nearest_large.copy()
    small=~np.isin(labels,list(large))
    score[small]=nearest_large[small] + d[small,labels[small]]

    return score, labels, sizes, centers

def add_cblof_features(df, feature_cols, n_clusters=4):
    Z=df[feature_cols].replace([np.inf,-np.inf],np.nan).fillna(0)
    scaler=StandardScaler()
    X=scaler.fit_transform(Z)
    scores,labels,sizes,centers=cblof_scores(X,n_clusters=n_clusters)
    out=df.copy()
    out["cblof_score"]=scores
    out["cblof_cluster"]=labels
    out["cblof_cluster_size"]=sizes[labels]
    return out, scaler

# ========================= SOLAR SCENARIO =========================

"""Preliminary rooftop-solar scenarios with explicit pricing assumptions."""


def solar_scenario(
    annual_kwh,
    capacity_kw,
    yield_kwh_per_kw=1400,
    cost_3kw=80000,
    cost_above_3kw=50000,
    subsidy_3kw=78000,
    tariff=7.0,
    self_consumption=0.8,
    maintenance_pct=0.01,
    daytime_consumption_share=0.45,
    export_rate=3.0,
    pricing_scenario="incremental_above_3kw",
):
    """Return a labelled scenario; verify policy and site inputs before use.

    The supplied project assumptions do not define costs below 3 kW. The
    dashboard therefore starts at 3 kW and exposes two alternative meanings
    for the above-3-kW rate instead of silently extrapolating a smaller system.
    """
    if annual_kwh < 0 or capacity_kw < 3:
        raise ValueError("annual_kwh must be non-negative and capacity_kw must be at least 3 kW")
    if pricing_scenario not in {"incremental_above_3kw", "all_capacity_above_3kw_rate"}:
        raise ValueError("pricing_scenario is not supported")

    generation = capacity_kw * yield_kwh_per_kw
    daytime_demand = annual_kwh * daytime_consumption_share
    onsite = min(generation * self_consumption, daytime_demand)
    export = max(generation - onsite, 0)

    if capacity_kw == 3:
        gross_cost = cost_3kw
    elif pricing_scenario == "incremental_above_3kw":
        gross_cost = cost_3kw + (capacity_kw - 3) * cost_above_3kw
    else:
        gross_cost = capacity_kw * cost_above_3kw

    subsidy = min(subsidy_3kw, gross_cost)
    net_cost = gross_cost - subsidy
    gross_savings = onsite * tariff + export * export_rate
    maintenance = gross_cost * maintenance_pct
    annual_net_savings = max(gross_savings - maintenance, 0)
    payback = net_cost / annual_net_savings if annual_net_savings else None
    return {
        "annual_consumption_kwh": annual_kwh,
        "capacity_kw": capacity_kw,
        "generation": generation,
        "on_site_consumption": onsite,
        "export": export,
        "gross_cost": gross_cost,
        "subsidy": subsidy,
        "net_cost": net_cost,
        "gross_savings": gross_savings,
        "maintenance": maintenance,
        "annual_net_savings": annual_net_savings,
        "payback": payback,
        "pricing_scenario": pricing_scenario,
        "status": "preliminary consumption-based scenario; verify technical and policy inputs",
    }

# ========================= MODEL TRAINING / ANALYSIS =========================



def main(input_path, output_dir):
    print("=" * 70)
    print("CONSUMER CONSUMPTION ML ANALYSIS STARTED")
    print("=" * 70)
    print(f"Input file: {input_path}")
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    figdir=output_dir/"figures"; figdir.mkdir(exist_ok=True)

    print("\n[1/8] Loading Excel data...")
    df,months=load_data(input_path)
    print(f"Loaded {len(df):,} records and {len(months)} monthly columns.")
    df,first7,last3=build_features(df,months)
    df,threshold=make_target(df)
    print(f"[2/8] Features created. High-consumption threshold: {threshold:,.2f}")

    # Train the two supervised baselines on the same held-out split.
    features=["CONTRACT_LOAD","TARRIF","FEEDER_NAME","VILLAGE_NAME",
              "SOLAR_CONSUMER","train_avg","train_std","cv"]
    numeric=["CONTRACT_LOAD","train_avg","train_std","cv"]
    categorical=["TARRIF","FEEDER_NAME","VILLAGE_NAME","SOLAR_CONSUMER"]

    X,y=df[features],df["high_consumption_target"]
    Xt,Xv,yt,yv=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)

    pre=ColumnTransformer([
        ("num",Pipeline([("imp",SimpleImputer(strategy="median")),
                         ("sc",StandardScaler())]),numeric),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                         ("oh",OneHotEncoder(handle_unknown="ignore"))]),categorical)
    ])

    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    experiments=[]

    knn_pipe=Pipeline([("pre",pre),("model",KNeighborsClassifier())])
    knn_grid={"model__n_neighbors":[5,9,15,21,31],
              "model__weights":["uniform","distance"],
              "model__p":[1,2]}
    gs=GridSearchCV(knn_pipe,knn_grid,cv=cv,scoring="f1",n_jobs=-1)
    print("[3/8] Training/tuning KNN (this may take some time)...")
    gs.fit(Xt,yt)
    best_knn=gs.best_estimator_
    for params,score in zip(gs.cv_results_["params"],gs.cv_results_["mean_test_score"]):
        experiments.append({"model":"KNN","experiment":str(params),"cv_f1":score})
    pred=best_knn.predict(Xv); prob=best_knn.predict_proba(Xv)[:,1]
    rows=[{"model":"KNN (tuned)","accuracy":accuracy_score(yv,pred),
           "precision":precision_score(yv,pred,zero_division=0),
           "recall":recall_score(yv,pred,zero_division=0),
           "f1":f1_score(yv,pred,zero_division=0),
           "roc_auc":roc_auc_score(yv,prob)}]
    joblib.dump(best_knn,output_dir/"knn_tuned.joblib")

    tree_pipe=Pipeline([("pre",pre),("model",DecisionTreeClassifier(random_state=42,class_weight="balanced"))])
    tree_grid={"model__max_depth":[3,5,7,10,None],
               "model__min_samples_leaf":[10,25,50,100]}
    gt=GridSearchCV(tree_pipe,tree_grid,cv=cv,scoring="f1",n_jobs=-1)
    print("[4/8] Training/tuning Decision Tree...")
    gt.fit(Xt,yt)
    best_tree=gt.best_estimator_
    for params,score in zip(gt.cv_results_["params"],gt.cv_results_["mean_test_score"]):
        experiments.append({"model":"Decision Tree","experiment":str(params),"cv_f1":score})
    pred=best_tree.predict(Xv); prob=best_tree.predict_proba(Xv)[:,1]
    rows.append({"model":"Decision Tree (tuned)","accuracy":accuracy_score(yv,pred),
                 "precision":precision_score(yv,pred,zero_division=0),
                 "recall":recall_score(yv,pred,zero_division=0),
                 "f1":f1_score(yv,pred,zero_division=0),
                 "roc_auc":roc_auc_score(yv,prob)})
    joblib.dump(best_tree,output_dir/"decision_tree_tuned.joblib")

    pd.DataFrame(rows).to_csv(output_dir/"model_metrics_tuned.csv",index=False)
    print("[5/8] Saving model metrics and hyperparameter experiments...")
    pd.DataFrame(experiments).sort_values("cv_f1",ascending=False).head(30).to_csv(
        output_dir/"hyperparameter_experiments.csv",index=False)

    # Save confusion matrices for the report.
    for title,model in [("KNN",best_knn),("Decision Tree",best_tree)]:
        pred=model.predict(Xv)
        cm=confusion_matrix(yv,pred)
        fig,ax=plt.subplots(figsize=(4.5,4))
        im=ax.imshow(cm)
        ax.set_title(f"{title} Confusion Matrix")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        for i in range(2):
            for j in range(2):
                ax.text(j,i,str(cm[i,j]),ha="center",va="center")
        fig.tight_layout(); fig.savefig(figdir/f"{title.lower().replace(' ','_')}_confusion_matrix.png",dpi=160)
        plt.close(fig)

    # Profile consumers with standardized behavior and load features.
    kfeatures=["train_avg","train_std","train_min","train_max","cv","CONTRACT_LOAD"]
    Z=df[kfeatures].replace([np.inf,-np.inf],np.nan).fillna(0)
    scaler=StandardScaler(); Zs=scaler.fit_transform(Z)

    cluster_rows=[]
    print("[6/8] Finding consumer clusters and PCA structure...")
    best_km=None
    for k in [2,3,4,5,6]:
        km=KMeans(n_clusters=k,n_init=20,random_state=42)
        labels=km.fit_predict(Zs)
        sil=silhouette_score(Zs,labels)
        cluster_rows.append({"k":k,"silhouette_score":sil})
        if best_km is None or sil>best_km[0]:
            best_km=(sil,km,labels)
    pd.DataFrame(cluster_rows).to_csv(output_dir/"clustering_metrics.csv",index=False)
    df["cluster"]=best_km[2]

    pca=PCA(n_components=2,random_state=42)
    Zp=pca.fit_transform(Zs)
    pca_df=pd.DataFrame({"PC1":Zp[:,0],"PC2":Zp[:,1],"cluster":df["cluster"]})
    pca_df.to_csv(output_dir/"pca_coordinates.csv",index=False)
    pd.DataFrame({
        "component":["PC1","PC2"],
        "explained_variance_ratio":pca.explained_variance_ratio_
    }).to_csv(output_dir/"pca_explained_variance.csv",index=False)

    fig,ax=plt.subplots(figsize=(7,5))
    for cl in sorted(pca_df.cluster.unique()):
        s=pca_df[pca_df.cluster==cl]
        ax.scatter(s.PC1,s.PC2,s=7,alpha=.35,label=f"Cluster {cl}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("PCA Projection of Consumer Profiles")
    ax.legend(); fig.tight_layout(); fig.savefig(figdir/"pca_clusters.png",dpi=160); plt.close(fig)

    # Rank unusual profiles against the cluster structure.
    print("[7/8] Calculating CBLOF anomaly scores...")
    cblof_features=["train_avg","train_std","train_min","train_max","cv","CONTRACT_LOAD"]
    cbdf,_=add_cblof_features(df,cblof_features,n_clusters=best_km[1].n_clusters)
    cbdf["cblof_percentile"]=cbdf["cblof_score"].rank(pct=True)
    cbdf["cblof_flag"]=np.where(cbdf["cblof_percentile"]>=.99,"CBLOF anomaly—review required","Not flagged")
    cbdf[["Consumer Name","CONTRACT_LOAD","train_avg","train_std","cv","cblof_score","cblof_cluster","cblof_cluster_size","cblof_percentile","cblof_flag"]].to_csv(
        output_dir/"cblof_anomalies.csv",index=False)

    # Keep the contract-load calculation as a transparent screening baseline.
    latest=months[-1]
    benchmark=df["CONTRACT_LOAD"]*8*31*.8
    df["benchmark_energy_latest_kwh"]=benchmark
    df["consumption_ratio_latest"]=df[latest]/benchmark.replace(0,np.nan)
    df["anomaly_flag_latest"]=np.select(
        [df["consumption_ratio_latest"]<.25,df["consumption_ratio_latest"]>1.5],
        ["Low-consumption anomaly—review required","High-consumption anomaly—review required"],
        default="Normal/insufficient evidence")
    df[["Consumer Name","CONTRACT_LOAD",latest,"benchmark_energy_latest_kwh","consumption_ratio_latest","anomaly_flag_latest"]].to_csv(
        output_dir/"anomaly_screen_latest.csv",index=False)

    print("[8/8] Creating final dataset summary...")
    pd.DataFrame({
        "Metric":["Records","Monthly columns","First-7-month high threshold","Missing monthly cells",
                  "Solar consumers","Non-solar consumers","Unique feeders","Unique villages",
                  "Selected K","Selected K silhouette","PC1 variance","PC2 variance","CBLOF flagged at 99th percentile"],
        "Value":[len(df),len(months),threshold,int(df[months].isna().sum().sum()),
                 int((df.SOLAR_CONSUMER=="Y").sum()),int((df.SOLAR_CONSUMER=="N").sum()),
                 df.FEEDER_NAME.nunique(),df.VILLAGE_NAME.nunique(),best_km[1].n_clusters,
                 best_km[0],pca.explained_variance_ratio_[0],pca.explained_variance_ratio_[1],
                 int((cbdf["cblof_flag"]!="Not flagged").sum())]
    }).to_csv(output_dir/"dataset_summary.csv",index=False)

    # ---------------- COLAB VISIBLE OUTPUT ----------------
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 70)

    metrics_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame({
        "Metric":["Records","Monthly columns","First-7-month high threshold","Missing monthly cells",
                  "Solar consumers","Non-solar consumers","Unique feeders","Unique villages",
                  "Selected K","Selected K silhouette","PC1 variance","PC2 variance","CBLOF flagged at 99th percentile"],
        "Value":[len(df),len(months),threshold,int(df[months].isna().sum().sum()),
                 int((df.SOLAR_CONSUMER=="Y").sum()),int((df.SOLAR_CONSUMER=="N").sum()),
                 df.FEEDER_NAME.nunique(),df.VILLAGE_NAME.nunique(),best_km[1].n_clusters,
                 best_km[0],pca.explained_variance_ratio_[0],pca.explained_variance_ratio_[1],
                 int((cbdf["cblof_flag"]!="Not flagged").sum())]
    })

    print("\nMODEL PERFORMANCE:")
    if display is not None:
        display(metrics_df.round(4))
    else:
        print(metrics_df.to_string(index=False))

    print("\nDATASET SUMMARY:")
    if display is not None:
        display(summary_df)
    else:
        print(summary_df.to_string(index=False))

    print("\nTOP CBLOF ANOMALIES:")
    anomaly_preview = cbdf.sort_values("cblof_score", ascending=False).head(20)[
        ["Consumer Name","CONTRACT_LOAD","train_avg","train_std","cv",
         "cblof_score","cblof_cluster","cblof_cluster_size",
         "cblof_percentile","cblof_flag"]
    ]
    if display is not None:
        display(anomaly_preview)
    else:
        print(anomaly_preview.to_string(index=False))

    print("\nLOW/HIGH CONSUMPTION ANOMALY SCREENING:")
    screen_preview = df[
        ["Consumer Name","CONTRACT_LOAD",latest,
         "benchmark_energy_latest_kwh","consumption_ratio_latest","anomaly_flag_latest"]
    ].loc[df["anomaly_flag_latest"] != "Normal/insufficient evidence"].head(20)
    if display is not None:
        display(screen_preview)
    else:
        print(screen_preview.to_string(index=False))

    print(f"\nResults saved in: {output_dir.resolve()}")
    print("Files created:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            print(" -", f.relative_to(output_dir))

    return {
        "data": df,
        "metrics": metrics_df,
        "summary": summary_df,
        "cblof_anomalies": cbdf,
        "output_dir": output_dir
    }

def run_from_colab(input_path=None, output_dir="results"):
    """Run the project easily from Google Colab.

    If input_path is not supplied, the function looks for an Excel file
    in /content. If exactly one .xlsx/.xls file is found, it is used.
    """
    if input_path is None:
        content_dir = Path("/content")
        excel_files = list(content_dir.glob("*.xlsx")) + list(content_dir.glob("*.xls"))

        if len(excel_files) == 1:
            input_path = str(excel_files[0])
            print(f"Using input file: {input_path}")
        elif len(excel_files) == 0:
            # Automatically open the Google Colab upload window.
            try:
                from google.colab import files

                print("No Excel file found in /content.")
                print("Please select your Excel dataset in the upload window...")

                uploaded = files.upload()

                uploaded_excel = [
                    Path("/content") / name
                    for name in uploaded.keys()
                    if Path(name).suffix.lower() in {".xlsx", ".xls"}
                ]

                if len(uploaded_excel) == 1:
                    input_path = str(uploaded_excel[0])
                    print(f"Uploaded input file: {input_path}")
                elif len(uploaded_excel) == 0:
                    raise FileNotFoundError(
                        "No Excel file was uploaded. Please upload an .xlsx or .xls file."
                    )
                else:
                    raise ValueError(
                        "Multiple Excel files were uploaded. Please upload only one "
                        "Excel dataset at a time."
                    )

            except ImportError:
                raise FileNotFoundError(
                    "No Excel file found in /content. Upload the Excel dataset "
                    "before running this script outside Google Colab."
                )
        else:
            files = "\n".join(str(f) for f in excel_files)
            raise ValueError(
                "Multiple Excel files found in /content. Please specify one, "
                "for example:\nrun_from_colab('/content/Hansot_adjusted_consumption.xlsx')\n\n"
                f"Files found:\n{files}"
            )

    return main(input_path, output_dir)


if __name__=="__main__":
    # Works both in normal Python/GitHub execution and Google Colab.
    # In Colab, Jupyter adds its own command-line arguments, so we use
    # parse_known_args() and make --input optional.
    ap=argparse.ArgumentParser(description="Consumer consumption ML analysis")
    ap.add_argument("--input", default=None, help="Path to input Excel file")
    ap.add_argument("--output", default="results", help="Output directory")
    a, _ = ap.parse_known_args()

    if a.input:
        main(a.input, a.output)
    else:
        run_from_colab(output_dir=a.output)
