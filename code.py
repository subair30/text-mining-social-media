
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

texts = [
    "Elon Musk founded Tesla",
    "Bill Gates founded Microsoft",
    "Sundar Pichai leads Google",
    "Amazon acquired Whole Foods",
    "Google develops AI tools",
    "I love machine learning",
    "Text mining is interesting",
    "Machine learning is powerful"
]

nlp = spacy.load("en_core_web_sm")

relations = []
entity_types = {}

for text in texts:
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    if len(entities) >= 2:
        subj = entities[0][0]
        obj = entities[1][0]

        entity_types[subj] = entities[0][1]
        entity_types[obj] = entities[1][1]

        verb = [t.text for t in doc if t.pos_ == "VERB"]
        verb = verb[0] if verb else "related"

        relations.append((subj, verb, obj))

print("\n===== CLEAN RELATIONS =====")
print(relations)

# GRAPH
G = nx.DiGraph()

for subj, verb, obj in relations:
    G.add_edge(subj, obj, label=verb)

plt.figure(figsize=(10,7))
pos = nx.spring_layout(G, k=1.5)

colors = []
for node in G.nodes():
    if entity_types.get(node) == "PERSON":
        colors.append("lightgreen")
    else:
        colors.append("skyblue")

nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=2500)
nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

plt.title("Entity Relationship Graph (Clean)")
plt.axis("off")
plt.show()


# BoW
bow = CountVectorizer(stop_words='english')
X_bow = bow.fit_transform(texts)
bow_df = pd.DataFrame(X_bow.toarray(), columns=bow.get_feature_names_out())

print("\n===== BAG OF WORDS =====")
print(bow_df)

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf.fit_transform(texts)
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())

print("\n===== TF-IDF =====")
print(tfidf_df)


bow_sum = bow_df.sum()
tfidf_sum = tfidf_df.sum()

df_compare = pd.DataFrame({
    "BoW": bow_sum,
    "TF-IDF": tfidf_sum
}).sort_values(by="TF-IDF", ascending=False).head(10)

df_compare.plot(kind="bar")
plt.title("BoW vs TF-IDF Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,5))
sns.heatmap(tfidf_df, cmap="viridis")
plt.title("TF-IDF Heatmap")
plt.show()

print("\n===== TOP TERMS =====")
features = tfidf.get_feature_names_out()

for i, row in enumerate(tfidf_df.values):
    top_idx = np.argsort(row)[-3:]
    print(f"Doc {i+1}:", [features[j] for j in top_idx])

word_freq = bow_df.sum().sort_values(ascending=False).head(10)

plt.figure()
word_freq.plot(kind='bar')
plt.title("Top Word Frequencies")
plt.xticks(rotation=45)
plt.show()



texts_cls = [
    "I love this product it is amazing",
    "This is the worst service ever",
    "Buy now click here limited offer",
    "Absolutely fantastic experience",
    "Very bad quality not recommended",
    "Win money now click here",
    "Great performance and excellent features",
    "Terrible support and poor response",
    "Free offer exclusive deal click now",
    "I am very happy with this service"
]

labels = [1,0,0,1,0,0,1,0,0,1]

X_cls = tfidf.fit_transform(texts_cls)

X_train, X_test, y_train, y_test = train_test_split(
    X_cls, labels, test_size=0.3, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n===== CLASSIFICATION =====")
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()