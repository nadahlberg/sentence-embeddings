from plotly.offline import plot
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import textwrap
import numpy as np 


class Embeddings:
	def __init__(self, vectors, texts=None, labels=None):
		self.vectors = vectors
		self.texts = texts
		self.labels = labels
		self.search_index = None

	@property
	def pca_3d(self):
		if not hasattr(self, '_pca_3d'):
			self._pca_3d = self._fit_pca(3)
		return self._pca_3d

	@property
	def pca_2d(self):
		if not hasattr(self, '_pca_2d'):
			self._pca_2d = self._fit_pca(2)
		return self._pca_2d

	def _fit_pca(self, n):
		print('Fitting PCA eith n=%i' % n)
		pca = PCA(n_components=n)
		pca.fit(self.vectors)
		return pca

	def plot_3d(self, save_path=None, show=True):
		if self.texts is not None:
			texts = ["<br>".join(textwrap.wrap(x)) for x in self.texts]
		else:
			texts = [""] * len(self.texts)
		
		labels = self.labels if self.labels is not None else [0] * len(self.texts)
		label_names = list(set(labels))

		points = self.pca_3d.transform(self.vectors)

		charts = []
		for label in label_names:
			label_points = [points[i] for i in range(len(points)) if labels[i] == label]
			label_texts = [texts[i] for i in range(len(texts)) if labels[i] == label]
			charts.append(
				go.Scatter3d(
					name=label,
					x=[x[0] for x in label_points], 
					y=[x[1] for x in label_points],
					z=[x[2] for x in label_points], 
					marker=dict(
						size=4,
						line=dict(
							width=.5, 
							color='DarkSlateGrey'
						)
					),
					hovertemplate= '%{text}',
					text=label_texts,
					mode='markers'
				)
			)

		fig = go.Figure(charts)
		fig = fig.update_layout(showlegend=False)

		if show:
			fig.show()
		if save_path is not None:
			plot(fig, filename=save_path)
		return fig

