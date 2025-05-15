import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
from preprocess import preprocess_dataframe

# Load and preprocess data
df = pd.read_csv('data/sample_dataset.csv')  # Columns: 'text', 'label'
df = preprocess_dataframe(df)

X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model pipelines
pipelines = {
    'logistic_regression': Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ]),
    'svm': Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', SVC())
    ]),
    'random_forest': Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', RandomForestClassifier())
    ])
}

# Train and save models
for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, f'models/{name}.pkl')
    print(f"{name} model trained and saved.")
