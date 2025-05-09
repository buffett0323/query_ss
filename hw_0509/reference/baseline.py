import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from matplotlib import pyplot as plt


# Custom dataset class
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# Model architecture
class ReviewClassifier(nn.Module):
    def __init__(self, bert_model, dropout=0.3):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.final_layer = nn.Sequential(
            nn.Linear(768, 1),  # 768 is BERT's hidden size
            nn.Sigmoid()
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Use [CLS] token representation
        dropout_output = self.dropout(pooled_output)
        return self.final_layer(dropout_output)
    



# Training function
def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].unsqueeze(1).float().to(device)

        # model forward
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = nn.BCELoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(data_loader)



# Evaluation function
def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].unsqueeze(1).float().to(device)
            
            # model forward
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = (outputs > 0.5).long().squeeze(1)
            
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.tolist())
            
    return predictions, actual_labels



# Main training loop
def main():
   
    # Hyperparameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LR = 2e-5 # 1e-4
    PATIENCE = 5  # Number of epochs to wait before early stopping
    
    # 1. Load and preprocess data
    df = pd.read_csv('review_data.csv', header=0, names=['id', 'review', 'helpfulness'])
    X, y = df['review'], df['helpfulness']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val = X_train.tolist(), X_val.tolist()
    y_train, y_val = y_train.tolist(), y_val.tolist()
    
    test_data = pd.read_csv('X_test.csv', header=0, names=['id', 'review'])
    X_test = test_data['review'].tolist()
    y_test = [0] * len(X_test)
    
    # 2. Initialize tokenizer and model
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert = BertModel.from_pretrained(model_name)

    # ReviewClassifier for BERT
    model = ReviewClassifier(bert_model=bert)
    
    
    # 3. Create datasets and dataloaders
    train_dataset = ReviewDataset(X_train, y_train, tokenizer)
    valid_dataset = ReviewDataset(X_val, y_val, tokenizer)
    test_dataset = ReviewDataset(X_test, y_test, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 4. Training setup
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # Early stopping setup
    best_accuracy = 0
    patience_counter = 0
    train_losses = []
    
    
    # 5. Training loop
    pbar = tqdm(range(NUM_EPOCHS), desc="Training")
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        predictions, actual_labels = evaluate(model, valid_loader, device)
        accuracy = sum(1 for x, y in zip(predictions, actual_labels) if x == y) / len(predictions)
        
        # Update progress bar description with metrics
        pbar.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {train_loss:.4f} | Acc: {accuracy:.4f}")
        
        # Early stopping check
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), 'baseline_best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
        
        
    # Plot training loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.close()
    
    
    # Load best model
    model.load_state_dict(torch.load('baseline_best_model.pt'))
    
    # Evaluate on test set
    predictions, _ = evaluate(model, test_loader, device)
    print(len(predictions))




if __name__ == "__main__":
    main()
