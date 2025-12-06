
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('dataset.csv')

tfidf = TfidfVectorizer()
X_text = tfidf.fit_transform(df['email_text'])

df['url_length'] = df['url'].apply(len)
X_url = df[['url_length']]

y = df['label']

X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)
X_train_url, X_test_url = train_test_split(X_url, test_size=0.2, random_state=42)

text_model = LogisticRegression()
text_model.fit(X_train_text, y_train)

url_model = RandomForestClassifier()
url_model.fit(X_train_url, y_train)

joblib.dump(text_model, 'text_model.pkl')
joblib.dump(url_model, 'url_model.pkl')
joblib.dump(tfidf, 'tfidf.pkl')

print("Training complete.")
