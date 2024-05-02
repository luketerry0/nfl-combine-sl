from preproccessing import Data
from sklearn import decomposition, preprocessing, cross_decomposition
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# load in data
data = Data.get_data(na_treatment="zeroed", proportions=[1])[0]
inputs, targets = data

print(targets)
print(sum(targets)/len(targets))
print(1 - (sum(targets)/len(targets)))

# scaler = preprocessing.StandardScaler()
# scaler.fit(inputs)
# Scaled_data=scaler.transform(inputs)

# pca = decomposition.PCA(n_components=10)
# pca.fit(Scaled_data)

# x=pca.transform(Scaled_data)


# plt.figure(figsize=(10,10))
# # plt.scatter(x[:,0],x[:,1],c=targets)
# # plt.xlabel('pc1')
# # plt.ylabel('pc2')
# # plt.show()

# # correlation analysis
# full_data = pd.DataFrame(np.hstack((inputs, np.array([targets]).T)))
# plt.matshow(full_data.corr())
# # plt.xticks(range(full_data.select_dtypes(['number']).shape[1]), full_data.select_dtypes(['number']).columns, fontsize=14, rotation=45)
# # plt.yticks(range(full_data.select_dtypes(['number']).shape[1]), full_data.select_dtypes(['number']).columns, fontsize=14)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.title('Correlation Matrix', fontsize=16)
# plt.show()
# # print(x)
# # full_data = pd.DataFrame(np.hstack((x,  np.array([targets]).T)))
# plt.matshow(full_data.corr())
# # plt.xticks(range(full_data.select_dtypes(['number']).shape[1]), full_data.select_dtypes(['number']).columns, fontsize=14, rotation=45)
# # plt.yticks(range(full_data.select_dtypes(['number']).shape[1]), full_data.select_dtypes(['number']).columns, fontsize=14)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.title('PCA Correlation Matrix', fontsize=16)
# plt.show()

