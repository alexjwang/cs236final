from sentence_transformers import SentenceTransformer
# nltk.download('wordnet')
# nltk.download('stopwords')
from sklearn.cluster import KMeans
import pandas as pd 

class Clusters():
	def __init__(self):
		self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
		self.num_clusters = 30

	def train(self, txt):
		corpus_embeddings = self.embedder.encode(txt.values)
		self.kmeans = KMeans(n_clusters=self.num_clusters,init='k-means++',max_iter=300,n_init=10,random_state=0)
		self.kmeans.fit(corpus_embeddings)
		return self.kmeans.predict(corpus_embeddings)


	def test(self, txt):
		corpus_embeddings = self.embedder.encode(txt.values)
		return self.kmeans.predict(corpus_embeddings)


train = pd.read_json('MathQA/train.json')[0:500]
train['text'] = 'Category : '+ train.category + '.# ' + train.Rationale.str.replace('"', '') + ' # ' + train.Problem
c = Clusters()
print(c.train(train['text']))
print(c.test(train['text']))
		