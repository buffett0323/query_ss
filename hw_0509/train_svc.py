from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.calibration import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from transformers import BertTokenizer, BertModel
import torch


model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)




def handle_embedding(doc):
    # Tokenize and encode the document
    inputs = tokenizer(doc, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state  # All token embeddings

    # Mean pooling to generate sentence embedding
    attention_mask = inputs['attention_mask']
    sentence_embedding = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1), dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)

    return sentence_embedding.squeeze()


if __name__ == "__main__":
    
    TEST_SIZE = 0.2 # 0.2
    df = pd.read_csv('review_data.csv', header=0, names=['id', 'review', 'helpfulness'])
    X, y = df['review'], df['helpfulness']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    X_train, X_val = X_train.tolist(), X_val.tolist()
    y_train, y_val = y_train.tolist(), y_val.tolist()
    
    test_data = pd.read_csv('X_test.csv', header=0, names=['id', 'review'])
    X_test = test_data['review'].tolist()
    y_test = [0] * len(X_test)
    
    
    X_train = [handle_embedding(doc) for doc in X_train]
    X_val = [handle_embedding(doc) for doc in X_val]
    X_test = [handle_embedding(doc) for doc in X_test]
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)
    
    # test data
    print("embedding train data")
    test_x_ids = [i for i in range(1, 1096) if i not in train_x_ids]
    test_x = [handle_embedding(docs[i-1]) for i in test_x_ids]
    print("test data loaded")

    # SVM
    SVM_model = SVC(
        kernel="linear",
        C=1,
        probability=True,
    )
    SVM_model.fit(train_x, train_y)

    # predict
    predictions = SVM_model.predict(test_x)
    df = pd.DataFrame({"Id": test_x_ids, "Value": predictions})
    df.to_csv("SVM_linear.csv", index=False)

    # Validate
    X_train, X_test, Y_train, Y_test = train_test_split(
        train_x, train_y, test_size=0.1, train_size=0.9, random_state=3, stratify=train_y
    )
    SVM_model.fit(X_train, Y_train)
    print(metrics.classification_report(SVM_model.predict(X_test), Y_test))

    # Curve
    y_score = SVM_model.predict_proba(X_test)
    Y_test = label_binarize(Y_test, classes=[*range(1, 14)])
    precision = {}
    recall = {}
    for i in range(1, 14):
        precision[i], recall[i], _ = metrics.precision_recall_curve(
            Y_test[:, i - 1], y_score[:, i - 1]
        )
        plt.plot(recall[i], precision[i], lw=2, label="class {}".format(i))

    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.show()