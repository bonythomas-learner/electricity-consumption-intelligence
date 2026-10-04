import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

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

from data_utils import load_data, build_features, make_target
from cblof import add_cblof_features

def main(input_path, output_dir):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    figdir=output_dir/"figures"; figdir.mkdir(exist_ok=True)

    df,months=load_data(input_path)
    df,first7,last3=build_features(df,months)
    df,threshold=make_target(df)

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

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True); ap.add_argument("--output",default="results")
    a=ap.parse_args(); main(a.input,a.output)
