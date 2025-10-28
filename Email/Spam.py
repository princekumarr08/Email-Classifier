import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("Email/email.csv")  
df = df[['Category', 'Message']]
df = df.dropna()
df['Category'] = df['Category'].str.strip().str.lower()
df['Message'] = df['Message'].astype(str)

print("Dataset loaded successfully!")
print("Sample data:\n", df.head(), "\n")
print("Number of rows after cleaning:", len(df), "\n")

X_train, X_test, y_train, y_test = train_test_split(
    df['Message'], df['Category'], test_size=0.2, random_state=42
)


vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)
print(" Model Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


while True:
    user_input = input("\n Enter an email/message to classify: ")
    if user_input.lower() == 'exit':
        print(" Exiting program...")
        break
    user_input_tfidf = vectorizer.transform([user_input])
    prediction = model.predict(user_input_tfidf)
    print(" Prediction:", prediction[0].upper())
