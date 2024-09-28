import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# Define the HealthcarePredictionModel class
class HealthcarePredictionModel(nn.Module):
    def __init__(self, input_features):
        super(HealthcarePredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.4)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
    
    def forward(self, x):
        x = self.bn1(torch.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(torch.relu(self.fc2(x)))
        x = self.dropout(x)
        x = self.bn3(torch.relu(self.fc3(x)))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna()
    categorical_columns = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_columns)
    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

# Main execution
if __name__ == "__main__":
    try:
        # Use raw string for file path
        file_path = r"C:\Users\Harshan\Documents\dsu hackothon\hospital_1_1data.csv"  # Replace with your actual data file path
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        print(f"Loading data from {file_path}")
        features, labels = load_and_preprocess_data(file_path)
        
        # Create dataset and dataloader
        dataset = TensorDataset(features, labels)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        # Initialize model, loss function, and optimizer
        input_features = features.shape[1]
        model = HealthcarePredictionModel(input_features)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            for batch_features, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    outputs = model(batch_features)
                    val_loss += criterion(outputs, batch_labels).item()
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == batch_labels).sum().item()
            
            val_loss /= len(val_loader)
            accuracy = correct / len(val_dataset)
            print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}')

        # Save the model
        model_save_path = os.path.join(os.path.dirname(__file__), 'trained_model.pth')
        torch.save({
            'input_features': input_features,
            'state_dict': model.state_dict()
        }, model_save_path)

        print(f"Model trained and saved successfully to {model_save_path}")

        # Verify the save was successful
        loaded_data = torch.load(model_save_path, map_location=torch.device('cpu'))
        print("Model loaded successfully for verification.")
        print(f"Input features: {loaded_data['input_features']}")
        print(f"State dict keys: {loaded_data['state_dict'].keys()}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()