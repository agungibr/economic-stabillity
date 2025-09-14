import torch
import torch.nn as nn
import joblib 
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from model.LSTM import LSTMModel

def create_sequences(features, labels, sequence_length):
    xs, ys = [], []
    for i in range(len(features) - sequence_length):
        x = features[i:(i + sequence_length)]
        y = labels[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def main():
    project_root = Path(__file__).parent
    res_folder = project_root / 'Data' / 'Res'
    model_folder = project_root / 'model'
    model_folder.mkdir(exist_ok=True)
    
    train_file = res_folder / 'train_preprocessed.csv'
    model_path = model_folder / 'lstm_model.pt'
    scaler_path = model_folder / 'scaler.pkl'
    
    SEQUENCE_LENGTH = 30
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    OUTPUT_SIZE = 3
    BATCH_SIZE = 8 
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001

    train_df = pd.read_csv(train_file)

    df_train = train_df[train_df['year'] < 2020].copy()
    df_val = train_df[train_df['year'] == 2020].copy()

    TARGET = 'economic_day_status'
    FEATURES = [col for col in df_train.columns if col not in [TARGET, 'year', 'id']]

    scaler = StandardScaler()
    df_train[FEATURES] = scaler.fit_transform(df_train[FEATURES])
    df_val[FEATURES] = scaler.transform(df_val[FEATURES])

    print(f"Creating sequences with a look-back of {SEQUENCE_LENGTH} days")
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

    for epoch in range(NUM_EPOCHS):
        model.train()
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        score = f1_score(all_labels, all_preds, average='macro')
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}, Validation Macro F1-Score: {score:.4f}')

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to '{model_path}'")
    print(f"Scaler saved to '{scaler_path}'")

if __name__ == '__main__':
    main()