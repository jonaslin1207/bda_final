# == Import necessary libraries ==
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.preprocessing import StandardScaler

# == Load the datasets ==
public_data_df = pd.read_csv("public_data.csv")
private_data_df = pd.read_csv("private_data.csv")

# == Extract the detector data columns ==
public_data = public_data_df.iloc[:, 1:5].values
private_data = private_data_df.iloc[:, 1:7].values

# == Determine the clustering method and scaling ==
scaling = "max"  # "max" or "z-score"
method = "kmeans"  # "kmeans", "agglomerative", "spectral"

# == Perform scaling ==
if scaling == "zscore":
    # Scale the data using Z-score normalization
    scaler_public = StandardScaler()
    scaler_private = StandardScaler()
    scaled_public_data = scaler_public.fit_transform(public_data)
    scaled_private_data = scaler_private.fit_transform(private_data)
elif scaling == "max":
    # Scale the data by dividing by the maximum value in each column
    scaled_public_data = public_data / np.max(public_data, axis=0)
    scaled_private_data = private_data / np.max(private_data, axis=0)

# Find the dimension of the data
n = scaled_public_data.shape[1]
m = scaled_private_data.shape[1]

# == Perform clustering ==
if method == "kmeans":
    kmeans_public = KMeans(n_clusters=4*n-1, init='k-means++', n_init=1, max_iter=2000, tol=1e-4, verbose=0, random_state=42, copy_x=True, algorithm='lloyd')
    kmeans_private = KMeans(n_clusters=4*m-1, init='k-means++', n_init=1, max_iter=2000, tol=1e-4, verbose=0, random_state=42, copy_x=True, algorithm='lloyd')
    publicData_result = kmeans_public.fit(scaled_public_data)
    privateData_result = kmeans_private.fit(scaled_private_data)
elif method == "agglomerative":
    agglemorative_public = AgglomerativeClustering(n_clusters=4*n-1, metric='euclidean', linkage='ward')
    agglemorative_private = AgglomerativeClustering(n_clusters=4*m-1, metric='euclidean', linkage='ward')
    publicData_result = agglemorative_public.fit(scaled_public_data)
    privateData_result = agglemorative_private.fit(scaled_private_data)
elif method == "spectral":
    spectral_public = SpectralClustering(n_clusters=4*n-1, affinity='nearest_neighbors', random_state=42, assign_labels='discretize')
    spectral_private = SpectralClustering(n_clusters=4*m-1, affinity='nearest_neighbors', random_state=42, assign_labels='discretize')
    publicData_result = spectral_public.fit(scaled_public_data)
    privateData_result = spectral_private.fit(scaled_private_data)

# == Save the results ==
public_result_df = pd.DataFrame(data={
    "id": public_data_df["id"],
    "label": publicData_result.labels_
})
private_result_df = pd.DataFrame(data={
    "id": private_data_df["id"],
    "label": privateData_result.labels_
})
# Save the results to CSV files
public_result_df.to_csv("r13b44015_public.csv", index=False)
private_result_df.to_csv("r13b44015_private.csv", index=False)
