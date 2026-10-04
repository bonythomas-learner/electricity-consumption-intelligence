"""Clustering-based anomaly scores used in the project review queue.

The implementation follows the CBLOF idea: K-Means forms behavioral groups,
cluster sizes define large and small groups, and distance to a large group
contributes to the review score. Keeping the calculation here makes the
assumptions visible and avoids an extra runtime dependency.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

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
