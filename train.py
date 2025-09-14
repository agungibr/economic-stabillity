import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from model.LSTM import LSTMModel

def create_sequences(features, labels, sequence_length):
    xs, ys = [], []
    for i in range(len(features) - sequence_length):
        x = features[i:(i + sequence_length)]
        y = labels[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def plot_training_history(epochs, losses, f1_scores, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, losses, label='Training Loss')
    ax1.set_title('Training Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, f1_scores, label='Validation Macro F1-Score', color='orange')
    ax2.set_title('Validation Macro F1-Score per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1-Score')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history chart saved to '{save_path}'")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix for Validation Set')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_path)
    print(f"Confusion matrix chart saved to '{save_path}'")


def main():
    project_root = Path(__file__).parent
    res_folder = project_root / 'Data' / 'Res'
    model_folder = project_root / 'model'
    model_folder.mkdir(exist_ok=True)
    
    train_file = res_folder / 'train_preprocessed.csv'
    model_path = model_folder / 'lstm_model.pt'
    scaler_path = model_folder / 'scaler.pkl'
    history_chart_path = model_folder / 'training_history.png'
    cm_chart_path = model_folder / 'confusion_matrix.png'
    
    SEQUENCE_LENGTH = 30
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    OUTPUT_SIZE = 3
    BATCH_SIZE = 64
    NUM_EPOCHS = 20 
    LEARNING_RATE = 0.001

    train_df = pd.read_csv(train_file)
    df_train = train_df[train_df['year'] < 2020].copy()
    df_val = train_df[train_df['year'] == 2020].copy()

    TARGET = 'economic_day_status'
    FEATURES = [col for col in df_train.columns if col not in [TARGET, 'year', 'id']]

    scaler = StandardScaler()
    df_train[FEATURES] = scaler.fit_transform(df_train[FEATURES])
    df_val[FEATURES] = scaler.transform(df_val[FEATURES])
    
    X_train, y_train = create_sequences(df_train[FEATURES].values, df_train[TARGET].values, SEQUENCE_LENGTH)
    X_val, y_val = create_sequences(df_val[FEATURES].values, df_val[TARGET].values, SEQUENCE_LENGTH)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\nStarting LSTM model training")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = LSTMModel(input_size=len(FEATURES), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    epoch_losses = []
    epoch_f1_scores = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        batch_losses = []
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        
        epoch_loss = np.mean(batch_losses)
        epoch_losses.append(epoch_loss)
        
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                outputs = model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        score = f1_score(all_labels, all_preds, average='macro')
        epoch_f1_scores.append(score)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}, Validation Macro F1-Score: {score:.4f}')

    print("\nFinal Evaluation on Validation Set")
    class_names = ['Low', 'Medium', 'High']
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    plot_training_history(range(1, NUM_EPOCHS + 1), epoch_losses, epoch_f1_scores, history_chart_path)
    plot_confusion_matrix(all_labels, all_preds, class_names, cm_chart_path)

if __name__ == '__main__':
    main()