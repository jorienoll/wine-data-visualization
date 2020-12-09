eimport numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets

raw_data = datasets.load_wine()

raw_data

print(raw_data['DESCR'])

print('data.shape\t',raw_data['data'].shape,
      '\ntarget.shape \t',raw_data['target'].shape)

features = pd.DataFrame(data=raw_data['data'],columns=raw_data['feature_names'])
data = features
data['target']=raw_data['target']

data.head()
#plots all
sns.set(style="ticks")
grid = sns.PairGrid(data, hue="target")
grid.map_diag(plt.hist)
grid.map_offdiag(plt.scatter)
grid.add_legend()

# plots alcohol vs nonflavanoid_phenols
a1 = data.loc[:,["alcohol","nonflavanoid_phenols","target"]]
sns.set(style="ticks")
grid = sns.PairGrid(a1, hue="target")
grid.map_diag(plt.hist)
grid.map_offdiag(plt.scatter)
grid.add_legend()

# plots total_phenols vs flavanoids
a2 = data.loc[:,["total_phenols","flavanoids","target"]]
sns.set(style="ticks")
grid = sns.PairGrid(a2, hue="target")
grid.map_diag(plt.hist)
grid.map_offdiag(plt.scatter)
grid.add_legend()

# get correlation coefficient of alcohol and nonflavanoid_phenols
data.groupby("target")[["alcohol","nonflavanoid_phenols"]].corr()

# get correlation coefficient of total_phenols and flavanoids
data.groupby("target")[["total_phenols","flavanoids"]].corr()

fig = plt.figure(figsize =(100, 100))
# plot alcohol box plots
data.boxplot(column=['alcohol'], by='target')
plt.title('alcohol');
plt.show()

fig = plt.figure(figsize =(100, 100))
# plot malic_acid box plots
data.boxplot(column=['malic_acid'], by='target')
plt.title('malic_acid');
plt.show()

fig = plt.figure(figsize =(100, 100))
# plot ash box plots
data.boxplot(column=['ash'], by='target')
plt.title('ash');
plt.show()

fig = plt.figure(figsize =(100, 100))
# plot alcalinity_of_ash box plots
data.boxplot(column=['alcalinity_of_ash'], by='target')
plt.title('alcalinity_of_ash');
plt.show()

fig = plt.figure(figsize =(100, 100))
# plot magnesium box plots
data.boxplot(column=['magnesium'], by='target')
plt.title('magnesium');
plt.show()

fig = plt.figure(figsize =(100, 100))
# plot total_phenols box plots
data.boxplot(column=['total_phenols'], by='target')
plt.title('total_phenols');
plt.show()

fig = plt.figure(figsize =(100, 100))
# plot flavanoids box plots
data.boxplot(column=['flavanoids'], by='target')
plt.title('flavanoids');
plt.show()

fig = plt.figure(figsize =(100, 100))
# plot nonflavanoid_phenols box plots
data.boxplot(column=['nonflavanoid_phenols'], by='target')
plt.title('nonflavanoid_phenols');
plt.show()

fig = plt.figure(figsize =(100, 100))
# plot proanthocyanins box plots
data.boxplot(column=['proanthocyanins'], by='target')
plt.title('proanthocyanins');
plt.show()

fig = plt.figure(figsize =(100, 100))
# plot color_intensity box plots
data.boxplot(column=['color_intensity'], by='target')
plt.title('color_intensity');
plt.show()

fig = plt.figure(figsize =(100, 100))
# plot hue box plots
data.boxplot(column=['hue'], by='target')
plt.title('hue');
plt.show()

fig = plt.figure(figsize =(100, 100))
# plot od/od box plots
data.boxplot(column=['od280/od315_of_diluted_wines'], by='target')
plt.title('od280/od315_of_diluted_wines');
plt.show()

fig = plt.figure(figsize =(100, 100))
# plot proline box plots
data.boxplot(column=['proline'], by='target')
plt.title('proline');
plt.show()
