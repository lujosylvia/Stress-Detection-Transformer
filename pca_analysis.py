import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
sns.set_theme(style="darkgrid")

#
# plots the explained variance data from PCA.
#
def plot_pca_scree(_pca: PCA):
  pca_df = pd.DataFrame(_pca.explained_variance_ratio_, columns=["Eigenvalues"])
  pca_df["Components"] = pca_df.index

  sns.lineplot(x="Components", y="Eigenvalues", data=pca_df)

#
# Performs PCA on a WESAD signal dataset.
# Defaults to using 4 components.
#
def pca_analysis(data_dict: dict, n_components: int = 4):
  _combined_df = pd.concat([data_dict[entry] for entry in data_dict.keys()], axis=0, ignore_index=True)

  _pca = PCA(n_components=n_components)
  components = _pca.fit_transform(_combined_df)

  pca_df = pd.DataFrame(data=components, columns=[f'PC{i+1}' for i in range(n_components)])
  plot_pca_scree(_pca)

  return pca_df