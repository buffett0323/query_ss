from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import torch

# Load BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)
bert_model.eval()  # set to evaluation mode

# Move model to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model = bert_model.to(device)

# Embedding function
def handle_embedding(doc):
    inputs = tokenizer(doc, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = bert_model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']

        # Mean pooling
        sentence_embedding = torch.sum(
            last_hidden_state * attention_mask.unsqueeze(-1), dim=1
        ) / attention_mask.sum(dim=1, keepdim=True)

    return sentence_embedding.squeeze().cpu().numpy()  # return numpy array


if __name__ == "__main__":
    TEST_SIZE = 0.2

    # Load data
    df = pd.read_csv('review_data.csv', header=0, names=['id', 'review', 'helpfulness'])
    X = df['review']
    y = df['helpfulness']
    X_train_text, X_val_text, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    # Embed data
    X_train_vec = np.stack([handle_embedding(doc) for doc in tqdm(X_train_text, desc="Embedding training data")])
    X_val_vec = np.stack([handle_embedding(doc) for doc in tqdm(X_val_text, desc="Embedding validation data")])

    # Load test data
    test_data = pd.read_csv('X_test.csv', header=0, names=['id', 'review'])
    X_test_text = test_data['review'].tolist()
    X_test_vec = np.stack([handle_embedding(doc) for doc in tqdm(X_test_text, desc="Embedding test data")])

    # Train SVM
    print("Training SVM...")
    SVM_model = SVC(kernel="linear", C=1, probability=True)
    SVM_model.fit(X_train_vec, y_train)

    # Predict on validation set
    print("Predicting on validation set...")
    predictions = SVM_model.predict(X_val_vec)
    accuracy = sum(1 for x, y in zip(predictions, y_val) if x == y) / len(predictions)
    print(f"Accuracy: {accuracy}")

    # Output predictions
    test_predictions = SVM_model.predict(X_test_vec)
    df_output = pd.DataFrame({
        "Id": test_data['id'],
        "Value": test_predictions
    })
    df_output.to_csv("SVM_linear.csv", index=False)
    print("Saved predictions to SVM_linear.csv")