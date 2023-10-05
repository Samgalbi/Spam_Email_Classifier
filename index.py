import os
import email
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load email data and labels from the SpamAssassin dataset
def load_data(folder):
    emails, labels = [], []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        with open(path, 'r', encoding='latin1') as file:
            msg = email.message_from_file(file)
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body += part.get_payload(decode=True).decode()
            else:
                body = msg.get_payload(decode=True).decode()
            emails.append(body)
            labels.append(1 if "spam" in filename else 0)  # 1 for spam, 0 for non-spam (ham)
    return emails, labels

# Load the dataset
data_folder = "Spam_Email_Classifier/dataset"


emails, labels = load_data(data_folder)

# TF-IDF feature extraction
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(emails)
tfidf_features = np.array(tfidf_matrix.toarray())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_features, labels, test_size=0.2, random_state=42)

# Train a Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate accuracy and generate a classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)
