from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))

texts = [x for x in data['data']][:500]
labels = [data['target_names'][x] for x in data['target']][:500]

vec = TfidfVectorizer()
vectors = vec.fit_transform(texts).toarray()

from sentence_embeddings import Embeddings

embeddings = Embeddings(vectors, texts=texts, labels=labels)

embeddings.plot_3d()
