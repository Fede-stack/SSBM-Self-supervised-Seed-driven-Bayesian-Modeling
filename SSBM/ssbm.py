import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.cluster import KMeans, DBSCAN
from time import sleep
import sklearn
import string
import nltk
from nltk.stem.porter import PorterStemmer
from krovetzstemmer import Stemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim
from gensim import corpora
from nltk.corpus import words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import operator
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from gensim.models.coherencemodel import CoherenceModel
import random
import math
import datetime
from nltk import bigrams
from collections import Counter
from itertools import permutations, combinations
import fasttext
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import adjusted_rand_score, confusion_matrix, homogeneity_completeness_v_measure
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import LabelSpreading
from sklearn.decomposition import PCA
import os
import keras
from keras.utils import to_categorical
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args



class SSBM:
    def __init__(self, docs, min_df=15, n_topics = 10, max_df=0.90, max_features=None, n_representations = 10, dimension=100, epoch=80, name='model'):
        self.docs = docs
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.dimension = dimension
        self.epoch = epoch
        self.name = name
        self.n_topics = n_topics
        self.n_representations = n_representations
        self.nlp_model = None
        self.binary_df = None
        self.words_to_consider = None
        self.pivot_table = None
        self.cluster_dict = None
        self.list_of_models = [LGBMClassifier(random_state=0), RandomForestClassifier(n_estimators=1000, random_state=0), MLPClassifier(hidden_layer_sizes=(512,), random_state=0)]

    def find_binary_representation(self):
        vectorizer = CountVectorizer(binary=True, min_df=self.min_df, max_df=self.max_df, max_features=self.max_features)
        binary_matrix = vectorizer.fit_transform(self.docs).toarray()
        self.binary_df = pd.DataFrame(binary_matrix, columns=vectorizer.get_feature_names_out())
        self.words_to_consider = np.array(list(self.binary_df.columns))
        return self.binary_df, self.words_to_consider

    def train_fasttext(self):
        file_path = f"corpus_{self.name}.txt"
        with open(file_path, "w", encoding="utf-8") as file:
            file.write("\n".join(self.docs))
        self.nlp_model = fasttext.train_unsupervised(file_path, dim=self.dimension, epoch=self.epoch)
        os.remove(file_path)
        return self.nlp_model

    def generate_word_embeddings(self):
        word_embeddings = [self.nlp_model.get_word_vector(word) for word in self.words_to_consider]
        return word_embeddings

    def create_pivot_table(self, word_embeddings):
        cosines = cosine_similarity(word_embeddings, word_embeddings)
        cosines = cosines-np.eye(cosines.shape[0])
        self.pivot_table = pd.DataFrame(cosines, index=self.words_to_consider, columns=self.words_to_consider)
        return self.pivot_table

    def find_seed_words(self, top_N, first_K):
        mean_df = pd.DataFrame()
        for index, row in self.pivot_table.iterrows():
            mean_value = row.nlargest(first_K).mean()
            mean_df.at[index, 'mean'] = mean_value
        sorted_mean_df = mean_df.sort_values(by='mean', ascending=False).head(top_N)
        seed_words = list(sorted_mean_df.index)
        return seed_words

    def find_clusters(self, seed_words):
        embeddings = self.pivot_table.loc[seed_words, seed_words]
        kmeans = KMeans(n_clusters=self.n_topics, random_state=0).fit(embeddings)
        cluster_labels = kmeans.labels_
        self.cluster_dict = {i: [] for i in range(kmeans.n_clusters)}
        for word, label in zip(seed_words, cluster_labels):
            self.cluster_dict[label].append(word)
        return self.cluster_dict
    
    def return_random_triplets(self, words, K, n_combinations):
        combinations_words = list(combinations(words, n_combinations))
        return random.sample(combinations_words, k=min(K, len(combinations_words)))

    def return_input2(self, triplet, word_to_consider):
        vector = np.zeros(len(word_to_consider))
        for word in triplet:
            vector[word_to_consider[word]] = 1
        return vector

    def prepare_data_for_modeling(self, K, n_combinations):
        input1 = []
        input2 = []
        out = []
        idx_corp = []

        word_to_consider = {word: idx for idx, word in enumerate(self.words_to_consider)}

        for doc_idx, doc in enumerate(self.docs):
            words_in_doc = set(doc.split())
            doc_vector = self.binary_df.iloc[doc_idx].values

            for cluster_id, cluster_words in self.cluster_dict.items():
                cluster_words_in_doc = set(cluster_words) & words_in_doc
                if len(cluster_words_in_doc) >= n_combinations:
                    random_triplets = self.return_random_triplets(list(cluster_words_in_doc), K, n_combinations)
                    for triplet in random_triplets:
                        idx_corp.append(doc_idx)
                        input1.append(doc_vector)
                        input2.append(self.return_input2(triplet, word_to_consider))
                        out.append(cluster_id)

        input1 = np.array(input1, dtype=np.float32)
        input2 = np.array(input2, dtype=np.float32)
        out = to_categorical(out, num_classes=len(self.cluster_dict))

        inputs_stack = np.column_stack((input1, input2))
        X_pred = np.eye(self.binary_df.shape[1])  # or whatever is the correct shape you intended

        return inputs_stack, out, X_pred

    def do_undersampling(self, c, inputs_stack, out):
        class_counts = Counter(out.argmax(axis=1))
        min_class_count = min(class_counts.values())
        desired_count = c * min_class_count

        sampling_strategy = {cls: min(count, desired_count) for cls, count in class_counts.items()}

        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = rus.fit_resample(inputs_stack, out)
        return X_resampled, y_resampled

    def extract_topic_representations(self, predictions, binary_df, n_representations):
        sorted_indices = np.argsort(predictions, axis=0)[::-1]
        top_10_indices = sorted_indices[:n_representations, :]
        topic_representations = [np.array(list(binary_df.columns))[list(top_10_indices[:, i_col])] for i_col in range(predictions.shape[1])]
        return topic_representations

    def calculate_topic_diversity(self, topics):
        all_words = [word for topic in topics for word in topic]
        total_words = len(all_words)
        unique_words = len(set(all_words))
        diversity_score = unique_words / total_words
        return diversity_score
    
    def get_predictions_scores(self, sel_mod, n_representations, X_resampled, y_resampled, X_pred,  binary_df):
        clf = self.list_of_models[sel_mod]
        clf.fit(X_resampled, np.argmax(y_resampled, axis=1))
        predictions = clf.predict_proba(np.column_stack((X_pred, X_pred)))
        topic_representations = self.extract_topic_representations(predictions, self.binary_df, n_representations)
        documents = [doc.split() for doc in self.docs]
        dictionary = corpora.Dictionary(documents)
        corpus = [dictionary.doc2bow(doc) for doc in documents]
        cm = CoherenceModel(topics=topic_representations, texts=documents, dictionary=dictionary, coherence='c_npmi')
        coherence_npmi = cm.get_coherence()
        cm = CoherenceModel(topics=topic_representations, texts=documents, dictionary=dictionary, coherence='c_uci')
        coherence_uci = cm.get_coherence()
        cm = CoherenceModel(topics=topic_representations, texts=documents, dictionary=dictionary, coherence='c_v')
        coherence_cv = cm.get_coherence()
        coherence_div = self.calculate_topic_diversity(topic_representations)
        prediction_clusters = clf.predict_proba(np.column_stack((np.zeros(binary_df.values.shape), binary_df.values)))
        return predictions, topic_representations, coherence_npmi, coherence_uci, coherence_cv, coherence_div, prediction_clusters

    def train(self, top_N, first_K, n_combinations, K, c, sel_mod, external_word_embeddings=None):
        self.find_binary_representation()
        if external_word_embeddings is None:
            self.train_fasttext()
            word_embeddings = self.generate_word_embeddings()
        else:
            word_embeddings = external_word_embeddings

        self.create_pivot_table(word_embeddings)
        seed_words = self.find_seed_words(top_N, first_K)
        self.find_clusters(seed_words)
        inputs_stack, out, X_pred = self.prepare_data_for_modeling(K, n_combinations)
        X_resampled, y_resampled = self.do_undersampling(c, inputs_stack, out)
        predictions, topic_representations, coherence_npmi, coherence_uci, coherence_cv, coherence_div, prediction_clusters = self.get_predictions_scores(sel_mod, 10, X_resampled, y_resampled, X_pred, self.binary_df)  
        return predictions, topic_representations, coherence_npmi, coherence_uci, coherence_cv, coherence_div, prediction_clusters
    
    def for_BO(self, top_N, first_K, n_combinations, K, c, sel_mod, word_embeddings):
        self.find_binary_representation()
        self.create_pivot_table(word_embeddings)  
        seed_words = self.find_seed_words(top_N, first_K)
        self.find_clusters(seed_words)
        inputs_stack, out, X_pred = self.prepare_data_for_modeling(K, n_combinations)
        X_resampled, y_resampled = self.do_undersampling(c, inputs_stack, out)
        predictions, topic_representations, coherence_npmi, coherence_uci, coherence_cv, coherence_div, prediction_clusters = self.get_predictions_scores(sel_mod, 10, X_resampled, y_resampled, X_pred, self.binary_df)  
        return -coherence_npmi

    def propagation_clusters(self, n_clusters,  topic_representations, model_name='all-miniLM-L6-v2'):
        model = SentenceTransformer(model_name)
        embeddings_se = model.encode(self.docs, show_progress_bar=True)

        li = [topic.tolist() for topic in topic_representations]
        embeddings_topics = np.array([model.encode(words) for words in li])

        pca = PCA(n_components=20)
        embs = pca.fit_transform(np.row_stack((embeddings_topics.mean(axis=1), embeddings_se)))
        embs_topics = embs[:n_clusters]
        embs = embs[n_clusters:]

        label_prop_model = LabelSpreading(alpha=.1, max_iter=60, n_jobs=-1)
        final_data = np.row_stack((embs_topics, embs))
        y_labs = np.concatenate((np.arange(n_clusters), np.repeat(-1, len(embeddings_se))))
        label_prop_model.fit(final_data, y_labs)
        predictions = label_prop_model.predict(final_data)[n_clusters:]
        return predictions
    
    def plot_topics(self, topic_representations):
        num_subplots = len(topic_representations)
        tt = [topics[:10] for topics in topic_representations]
        texts = [' \n '.join(text) for text in tt]
        num_cols = 5
        num_rows = -(-num_subplots // num_cols)  
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
        for i, ax in enumerate(axs.flatten()):
            if i < len(texts):
                ax.text(0.5, 0.5, texts[i], fontsize=12, ha='center', va='center',
                      bbox=dict(facecolor='black', edgecolor='black'), color = 'white')
                ax.axis('off')
            else:
                ax.axis('off')
        plt.show()
