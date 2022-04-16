import pandas as pd
# df = pd.read_json("https://raw.githubusercontent.com/nasa-petal/data-collection-and-prep/main/golden.json")
df = pd.read_json("golden.json")

# Do k-means clustering of abstracts
# https://pythonprogramminglanguage.com/kmeans-text-clustering/

# very good but a little long http://brandonrose.org/clustering

# docs for Kmeans https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score

df_nolabel = df[df['level1'].astype(bool)]
df_label = df[~ df['level1'].astype(bool)]

# list_text = df_nolabel[df_nolabel.level1].abstract.tolist()
not_labeled_abstracts = df_nolabel.abstract.tolist()
print(len(not_labeled_abstracts))
labeled_abstracts = df_label.abstract.tolist()
print(len(labeled_abstracts))

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(not_labeled_abstracts)



Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
    km = model.fit(X)
    Sum_of_squared_distances.append(km.inertia_)

import matplotlib.pyplot as plt
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


true_k = 7
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)

# model = MiniBatchKMeans(
#     n_clusters=true_k,
#     init="k-means++",
#     n_init=1,
#     init_size=1000,
#     batch_size=1000,
# )

model.fit(X)


print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

print("\n")
print("Prediction")

Y = vectorizer.transform(["chrome browser to open."])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["My cat is hungry."])
prediction = model.predict(Y)
print(prediction)

