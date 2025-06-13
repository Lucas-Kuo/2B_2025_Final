from cluster import get_cluster
import pandas as pd
from grader import score
from matplotlib import pyplot as plt
from tqdm import tqdm

from sklearn.cluster import KMeans

def kmeans_score(X, random_state=2000, initialization='random'):
    kmeans = KMeans(n_clusters=4*X.shape[1]-1, random_state=random_state, init=initialization)
    kmeans.fit(X)
    pred = kmeans.labels_
    return score(pred.tolist())

df = pd.read_csv("public_data.csv")
X = df.iloc[:, 1:].values
# normalize each dimension
X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)

no_normalized_scores = []
kmeans_pp_scores = []
random_scores = []
no_normalized_random_scores = []
for rand_state in tqdm(range(2000, 2100)):
    no_normalized_score = kmeans_score(X, random_state=rand_state, initialization='k-means++')
    no_normalized_scores.append(no_normalized_score)

    kmeans_pp_score = kmeans_score(X_normalized, random_state=rand_state, initialization='k-means++')
    kmeans_pp_scores.append(kmeans_pp_score)

    random_score = kmeans_score(X_normalized, random_state=rand_state, initialization='random')
    random_scores.append(random_score)

    no_normalized_random_score = kmeans_score(X, random_state=rand_state, initialization='random')
    no_normalized_random_scores.append(no_normalized_random_score)

# print average scores
print(f"Average no normalization score: {sum(no_normalized_scores) / len(no_normalized_scores)}")
print(f"Average k-means++ score: {sum(kmeans_pp_scores) / len(kmeans_pp_scores)}")
print(f"Average random score: {sum(random_scores) / len(random_scores)}")
print(f"Average no normalization random score: {sum(no_normalized_random_scores) / len(no_normalized_random_scores)}")

pred = get_cluster(X_normalized)
# store the final answer
# id, label
df_ans = pd.DataFrame({
    'id': range(1, len(pred) + 1),
    'label': pred
})
df_ans.to_csv('public_submission.csv', index=False, header=True)

# plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=pred, cmap='viridis', marker='o', s=10)
plt.title('KMeans Clustering of Public Data')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig('kmeans_clustering_12.png')
plt.close()

# plot the clusters on dimensions 2 and 3
plt.scatter(X[:, 1], X[:, 2], c=pred, cmap='viridis', marker='o', s=10)
plt.title('KMeans Clustering of Public Data (Dimensions 2 and 3)')
plt.xlabel('Dimension 2')
plt.ylabel('Dimension 3')
plt.savefig('kmeans_clustering_23.png')
plt.close()

# plot the clusters on dimensions 3 and 4
plt.scatter(X[:, 2], X[:, 3], c=pred, cmap='viridis', marker='o', s=10)
plt.title('KMeans Clustering of Public Data (Dimensions 3 and 4)')
plt.xlabel('Dimension 3')
plt.ylabel('Dimension 4')
plt.savefig('kmeans_clustering_34.png')
plt.close()

# plot the clusters on dimensions 1 and 4
plt.scatter(X[:, 0], X[:, 3], c=pred, cmap='viridis', marker='o', s=10)
plt.title('KMeans Clustering of Public Data (Dimensions 1 and 4)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 4')
plt.savefig('kmeans_clustering_14.png')
plt.close()

# correlation matrix
import seaborn as sns
corr = pd.DataFrame(X).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix of Public Data')
plt.savefig('correlation_matrix.png')
plt.close()

df = pd.read_csv("private_data.csv")
X = df.iloc[:, 1:].values

kmeans = KMeans(n_clusters=4*X.shape[1]-1, random_state=42, init="k-means++")
kmeans.fit(X)
pred = kmeans.labels_.tolist()

# store the final answer
df_ans = pd.DataFrame({
    'id': range(1, len(pred) + 1),
    'label': pred
})
df_ans.to_csv('private_submission.csv', index=False, header=True)