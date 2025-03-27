import streamlit as st
import torch
import joblib
import numpy as np
import torch 
import torch.nn as nn

# One-class Model
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Multiclass Model
class MultiClassClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=5):
        super(MultiClassClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)  # 5 output classes
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # No sigmoid, use raw logits for CrossEntropyLoss
        return x

# Initialize Model
input_dim = 790
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
one_class_model = BinaryClassifier(input_dim)
model_save_path = "./models/cyber_threat_one_class_classifier.pth"
# Recreate the model architecture
one_class_model = BinaryClassifier(input_dim=input_dim)  # Same architecture as before
one_class_model.load_state_dict(torch.load(model_save_path, map_location=device))
one_class_model.to(device)
one_class_model.eval()
vectorizer_one = joblib.load("./models/tfidf_vectorizer_one_class_classifier.pkl")

# Load Multiclass model and vectorizer
multi_class_model = MultiClassClassifier(input_dim=input_dim, num_classes=5)
multi_class_model.load_state_dict(torch.load("./models/multiclass_classifier.pth", map_location=device))
multi_class_model.to(device)
multi_class_model.eval()
vectorizer_multi = joblib.load("./models/tfidf_vectorizer_multiclass.pkl")

# Function for one-class inference
def predict_one_class(tweet):
    vector = vectorizer_one.transform([tweet]).toarray()
    tensor = torch.tensor(vector, dtype=torch.float32)

    with torch.no_grad():
        output = one_class_model(tensor).squeeze().item()

    label = "Normal" if output < 0.5 else "Threat"
    confidence = output if output >= 0.5 else 1 - output
    return label, confidence

# Function for multiclass inference
def predict_multiclass(tweet):
    vector = vectorizer_multi.transform([tweet]).toarray()
    tensor = torch.tensor(vector, dtype=torch.float32)

    with torch.no_grad():
        output = multi_class_model(tensor)
        _, predicted = torch.max(output, 1)

    label_names = ['Phishing', 'Malware', 'DDoS', 'Ransomware']
    label = label_names[predicted.item()]
    return label

# Streamlit UI
st.title("🚀 Cyber Threat Classifier")

tweet = st.text_area("Enter a tweet for classification:")

if st.button("Classify"):
    if tweet:
        # Step 1: One-class classification
        one_class_label, confidence = predict_one_class(tweet)

        # Display One-class result
        st.subheader("🔎 One-class Classification Result")
        st.write(f"**Label:** {one_class_label}")
        st.write(f"**Confidence:** {confidence:.4f}")

        if one_class_label == "Threat":
            # Step 2: Multiclass classification
            multi_label = predict_multiclass(tweet)

            st.subheader("⚠️ Threat Classification Result")
            st.write(f"**Threat Type:** {multi_label}")
        else:
            st.success("✅ The tweet is classified as **normal**.")
    else:
        st.error("⚠️ Please enter a tweet to classify.")
