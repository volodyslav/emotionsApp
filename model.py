from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib

df = pd.read_csv("text.csv")
df.isnull().sum()
df = df.drop(columns=["Unnamed: 0"], axis=0)
X = df["text"]
y = df["label"]



X_train, X_test, y_train, y_test = train_test_split(X, y)

tokenizer = TfidfVectorizer()
X_train_vect = tokenizer.fit_transform(X_train)
X_test_vect = tokenizer.transform(X_test)

cls_reglog = LogisticRegression()
cls_reglog.fit(X_train_vect, y_train)

y_pred = cls_reglog.predict(X_test_vect)
joblib.dump(cls_reglog, 'logistic_regression_model.pkl')
joblib.dump(tokenizer, 'tokenizer.pkl')
accuracy_logreg = accuracy_score(y_test, y_pred)
print("Accuracy Logistic Regression: ", accuracy_logreg)
classification_logreg = classification_report(y_test, y_pred)
print('Classification report Logistic Regression:')
print(classification_logreg)

