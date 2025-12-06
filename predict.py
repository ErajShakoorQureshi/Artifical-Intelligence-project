
import pandas as pd
import joblib

text_model = joblib.load('text_model.pkl')
url_model = joblib.load('url_model.pkl')
tfidf = joblib.load('tfidf.pkl')

def predict(email_text, url):
    text_vec = tfidf.transform([email_text])
    url_features = [[len(url)]]

    text_pred = text_model.predict_proba(text_vec)[0][1]
    url_pred = url_model.predict_proba(url_features)[0][1]

    final_score = (text_pred + url_pred) / 2
    return "Phishing" if final_score > 0.5 else "Legit"

print(predict("Your password expires now", "http://verify-now.com"))
