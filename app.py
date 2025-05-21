from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

nltk.download('stopwords')

app = Flask(__name__)

def load_data(filepath):
    df = pd.read_csv(filepath, header=None)
    df.columns = ['id', 'platform', 'emotion', 'text']
    df = df[['emotion', 'text']].dropna()
    return df

class CustomLabelEncoder:
    def fit(self, y):
        self.classes_ = np.unique(y)
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes_)}
        return self

    def transform(self, y):
        return np.array([self.class_to_index[cls] for cls in y])

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        return np.array([self.classes_[idx] for idx in y])

class CustomTfidfVectorizer:
    def __init__(self, stop_words):
        self.stop_words = set(stop_words)
        self.vocab = {}
        self.idf = {}

    def fit(self, X):
        total_docs = len(X)
        word_count = defaultdict(int)

        for text in X:
            words = set(text.lower().split())
            words = words - self.stop_words
            for word in words:
                word_count[word] += 1

        self.vocab = {word: idx for idx, word in enumerate(word_count.keys())}
        self.idf = {word: np.log(total_docs / count) for word, count in word_count.items()}

        return self

    def transform(self, X):
        result = lil_matrix((len(X), len(self.vocab)))

        for i, text in enumerate(X):
            words = text.lower().split()
            words = [word for word in words if word not in self.stop_words]
            word_freq = defaultdict(int)
            for word in words:
                if word in self.vocab:
                    word_freq[word] += 1
            for word, freq in word_freq.items():
                result[i, self.vocab[word]] = (freq / len(words)) * self.idf[word]

        return result.tocsr()

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class CustomMultinomialNB:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.class_count_ = np.zeros(len(self.classes_), dtype=np.float64)
        self.feature_count_ = np.zeros((len(self.classes_), X.shape[1]), dtype=np.float64)

        for i, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            self.class_count_[i] = X_cls.shape[0]
            self.feature_count_[i, :] = np.sum(X_cls, axis=0)

        self.feature_log_prob_ = np.log((self.feature_count_ + 1) / (self.feature_count_.sum(axis=1, keepdims=True) + X.shape[1]))
        self.class_log_prior_ = np.log(self.class_count_ / self.class_count_.sum())

        return self

    def predict(self, X):
        jll = X @ self.feature_log_prob_.T + self.class_log_prior_
        return self.classes_[np.argmax(jll, axis=1)]

def train_model(filepath):
    df = load_data(filepath)
    X = df['text']
    y = df['emotion']

    stop_words = list(stopwords.words('english'))

    le = CustomLabelEncoder()
    y = le.fit_transform(y)

    vectorizer = CustomTfidfVectorizer(stop_words)
    X_transformed = vectorizer.fit_transform(X)

    model = CustomMultinomialNB()
    model.fit(X_transformed, y)

    return model, le, vectorizer, df, y


model, label_encoder, vectorizer, df, y = train_model('./training.csv')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    if request.method == 'POST':
        text1 = request.form['text1']
        X_transformed = vectorizer.transform([text1])
        prediction = model.predict(X_transformed)[0]
        emotion = label_encoder.inverse_transform([prediction])[0]
        return render_template('analysis.html', final=emotion, text1=text1)
    return render_template('analysis.html')

@app.route('/statistics')
def statistics():
    X_transformed = vectorizer.transform(df['text'])
    y_pred = model.predict(X_transformed)
    y_true = label_encoder.inverse_transform(y)
    y_pred = label_encoder.inverse_transform(y_pred)

    confusion_mtx = confusion_matrix(y_true, y_pred)
    classification_rep = classification_report(y_true, y_pred, output_dict=True)
    accuracy = accuracy_score(y_true, y_pred)

    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('statistics.html', plot_url=plot_url, classification_rep=classification_rep, accuracy=accuracy)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
