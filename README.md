# Dependencies

This project was developed using Python 3.9.7. It uses the following libraries (all installed with `pip`):
- NumPy 1.20.3
- SciPy 1.7.1
- Matplotlib 3.4.3
- scikit-learn 0.24.2

The figures in the report document are sourced from the `examples.ipynb` Jupyter notebook, which can be viewed using the Jupyter server or VSCode's Jupyter plugin (among other alternatives).

# Usage

Given an $N$-dimensional dataset in the form of an $M\times N$ numpy array:
- Apply dimensionality reduction, e.g. t-SNE, to obtain an $M \times 2$ array.

```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)

data_tsne = tsne.fit_transform(data)
```

- Compute the Voronoi diagram.

```python
from scipy.spatial import Voronoi
data_tsne_vor = Voronoi(data_tsne)
```

- Render the crack/strain image.

```
from lib import *
fig,ax = plt.subplots()
draw_main(ax, data_tsne_vor, data, f_standard_cracks, f_standard_strain, beta=10, gamma=1)
plt.show()
```