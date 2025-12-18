from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def test_model_improvement():

    df = pd.read_csv("sentiments.csv")
    df["label"] = (df["sentiment"] + 1) // 2
    df = df.dropna(subset=["text", "label"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)

    print("\nAccuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, preds))

    assert acc > 0.5

test_model_improvement()
