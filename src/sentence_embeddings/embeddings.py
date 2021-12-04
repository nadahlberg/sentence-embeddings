from plotly.offline import plot
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import textwrap
import faiss


class Embeddings:
    def __init__(self, vectors, texts=None, labels=None):
        self.vectors = vectors
        self.texts = texts
        self.labels = labels

    @property
    def search_index(self):
        if not hasattr(self, '_search_index'):
            self._search_index = self._build_search_index()
        return self._search_index

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

    def _build_search_index(self):
        print('Creating search index')
        search_index = faiss.IndexFlatL2(self.vectors.shape[1])
        search_index.add(self.vectors.astype('float32'))
        return search_index

    def _fit_pca(self, n):
        print('Fitting PCA eith n=%i' % n)
        pca = PCA(n_components=n)
        pca.fit(self.vectors)
        return pca

    def search(self, query_vectors, k=1):
        D, I = self.search_index.search(query_vectors.astype('float32'), k)
        results = []
        for i1 in range(len(query_vectors)):
            results.append([])
            for i2 in range(k):
                result = {
                    'index': I[i1][i2],
                    'distance': D[i1][i2],
                }
                if self.texts is not None:
                    result['text'] = self.texts[I[i1][i2]]
                if self.labels is not None:
                    result['label'] = self.labels[I[i1][i2]]

                results[-1].append(result)
        return results


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

