import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF,LatentDirichletAllocation
from sklearn.manifold import TSNE
import joblib



df = pd.read_csv('bbc-text.csv',encoding='latin1')

null_values = df.isnull().sum()

#Text Preprocessing
def lowercasing(txt):
    return txt.lower()

def remove_stopwords(txt):
    stop_words = set(stopwords.words('english'))
    try:
        words = word_tokenize(txt)
    except TypeError as e:
        print(f'Error occured at remove stopwords {e}')
        return ""
    
    cleaned_words = [word for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

def remove_punctuation(txt):
    return txt.translate(str.maketrans('','',string.punctuation))

def remove_numbers(txt):
    return "".join([i for i in txt if not i.isdigit()])


df['text'] = df['text'].apply(lowercasing)
df['text'] = df['text'].apply(remove_numbers)
df['text'] = df['text'].apply(remove_punctuation)
df['text'] = df['text'].apply(remove_stopwords)

#Data preprocessing

x = df['text']
y = df['category']

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.33)

n_topics = 5
n_top_words = 10

vectorizer_tf = TfidfVectorizer()
vectorizer_bow = CountVectorizer()

x_train_tf = vectorizer_tf.fit_transform(x_train)
x_test_tf = vectorizer_tf.transform(x_test)

x_train_bow = vectorizer_bow.fit_transform(x_train)
x_test_bow = vectorizer_bow.transform(x_test)


#Helper to get words from NMF AND LDA
def get_top_words(model,feature_names,n_top_words):
    topics = []
    for topic in model.components_:
        top_words_indices = topic.argsort()[:-n_top_words - 1:-1]
        topics.append(" ".join([feature_names[i] for i in top_words_indices]))
    return topics    

#Model Training and Evaluation , Vectorization

#KMeans

kmeans = KMeans(n_clusters=n_topics,n_init='auto',random_state=42)
kmeans.fit(x_train_tf)

kmeans_feature_names = vectorizer_tf.get_feature_names_out()
kmeans_label = kmeans.labels_

print('Topic found by K-Means :')
for i in range(n_topics):
    cluster_indices = np.where(kmeans_label == i)[0]
    cluster_tdidf_sum = np.array(x_train_tf[cluster_indices].sum(axis=0)).flatten()
    top_words_indices =  cluster_tdidf_sum.argsort()[:-n_top_words - 1 : -1]
    top_words = [kmeans_feature_names[idx] for idx in top_words_indices]
    print(f'Topic # {i+1} :' " ".join(top_words) )

print('-' * 50)


#NMF 
topics_count_nmf = 10

nmf_model = NMF(n_components=topics_count_nmf,random_state=42,init='nndsvd',l1_ratio=0.5)
nmf_model.fit(x_train_tf)
nmf_feature_names = vectorizer_tf.get_feature_names_out()
nmf_topics = get_top_words(nmf_model,nmf_feature_names,n_top_words)

print('Topic found by NMF \n')

for i,topic in enumerate(nmf_topics):
    print(f"Topic # {i+1} : {topic}")

print('-' * 50)


#LDA (WITH Bag-of-words)
log_likelihoods = []
n_topics_lda = 7
lda_model = LatentDirichletAllocation(n_components=n_topics_lda,max_iter=10,learning_method='online',random_state=42)
lda_model.fit(x_train_bow)
log_likelihood = lda_model.score(x_train_bow)
log_likelihoods.append(log_likelihood)
print(f'Lda with {n_topics_lda} topic has a log-likelihood of : {log_likelihood}')




print('-' * 50)

#Finding the optimal numbers of clusters with Elbow Method

inertia = []
max_clusters = 20
for i in range(1,max_clusters+1):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,random_state=42)
    kmeans.fit(x_train_tf)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, max_clusters + 1), inertia, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

#Visualize the clusters

kmeans_labels = kmeans.predict(x_train_tf)

tsne_model = TSNE(n_components=2,random_state=42,perplexity=30.0,learning_rate='auto',init='pca')
tsne_data = tsne_model.fit_transform(x_train_tf.toarray())


plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=kmeans_labels, cmap='viridis', s=10)
plt.title('t-SNE visualization of K-Means clusters')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()

#Visualize the clusters
kmeans_labels = kmeans.predict(x_train_tf)
tsne_model = TSNE(n_components=2, random_state=42, perplexity=30.0, learning_rate='auto', init='pca')
tsne_data = tsne_model.fit_transform(x_train_tf.toarray())
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=kmeans_labels, cmap='viridis', s=10)
plt.title('t-SNE visualization of K-Means clusters')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()




print('Null Values :-', null_values)


best_model = nmf_model
vectorizer = vectorizer_tf

joblib.dump(best_model,'NMF_Model.joblib')
joblib.dump(vectorizer,'Vectorizer.joblib')

