import pandas as pd
import joblib
from sklearn.metrics import classification_report
from preprocess import preprocess_dataframe

# Load data
df = pd.read_csv('data/sample_dataset.csv')
df = preprocess_dataframe(df)

X = df['clean_text']
y = df['label']

# Load and evaluate models
model_names = ['logistic_regression', 'svm', 'random_forest']

for name in model_names:
    model = joblib.load(f'models/{name}.pkl')
    y_pred = model.predict(X)
    print(f"Evaluation for {name}:\n")
    print(classification_report(y, y_pred))
    print("="*60)
