# %%
### In this notebook I wanted to try and decrease the training time but not lose any accuracy
### The idea came from Leslie N. Smith and Nicholay Topin
### Super-convergence: very fast training of neural networks using large learning rates

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CyclicLR

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

from scipy.stats import zscore

pd.set_option('display.max_columns', None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
def rmsle(y_pred, y_true):
    return torch.sqrt(torch.mean(torch.square(torch.log(y_pred + 1) - torch.log(y_true + 1))))

def detect_outliers_and_create_feature(data, columns, threshold=3):
    for col in columns:
        # Calculate Z-scores for the selected column
        data[col + '_zscore'] = zscore(data[col])
        
        # Create a new binary feature indicating outliers
        data[col + '_is_outlier'] = (np.abs(data[col + '_zscore']) > threshold).astype(int)
        
    return data

# %%
test=pd.read_csv('./test.csv')
train=pd.read_csv('./train.csv')

enc=OneHotEncoder(drop='first')

test_enc=enc.fit_transform(test[['Sex']])
test_encoded_df = pd.DataFrame(test_enc.toarray(), columns=enc.get_feature_names_out(['Sex']))
test_df = pd.concat([test, test_encoded_df], axis=1)
test_df=test_df.drop(columns=['Sex'])

train_enc=enc.fit_transform(train[['Sex']])
train_encoded_df = pd.DataFrame(train_enc.toarray(), columns=enc.get_feature_names_out(['Sex']))
train_df = pd.concat([train, train_encoded_df], axis=1)
train_df=train_df.drop(columns=['Sex'])

# %%
# train_df=detect_outliers_and_create_feature(train_df,columns=['Length', 'Diameter', 'Height', 'Whole weight', 'Whole weight.1',
#        'Whole weight.2', 'Shell weight'],threshold=3)

# test_df=detect_outliers_and_create_feature(test_df,columns=['Length', 'Diameter', 'Height', 'Whole weight', 'Whole weight.1',
#        'Whole weight.2', 'Shell weight'],threshold=3)

# %%
train_df.head()

# %%
feature_names=[
    'Length',
    'Diameter', 
    'Height', 
    'Whole weight', 
    'Whole weight.1',
    'Whole weight.2', 
    'Shell weight', 
    'Sex_I', 
    'Sex_M',
       ]

X=train_df[feature_names].values
y=train_df.Rings.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)

X_train=torch.tensor(X_train, dtype=torch.float32).to(device)
y_train=torch.tensor(y_train, dtype=torch.float32).to(device)

X_test=torch.tensor(X_test, dtype=torch.float32).to(device)
y_test=torch.tensor(y_test, dtype=torch.float32).to(device)

# Convert data to PyTorch tensors and move them to GPU
X_train_cpu = X_train.clone().detach().to('cpu')
y_train_cpu = y_train.clone().detach().to('cpu')
X_test_cpu = X_test.clone().detach().to('cpu')
y_test_cpu = y_test.clone().detach().to('cpu')

X_cpu=X.copy()
y_cpu=y.copy()
X_gpu=torch.tensor(X, dtype=torch.float32).to(device)
y_gpu=torch.tensor(y, dtype=torch.float32).to(device)


# %%
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-Fold CV
fold_scores = []

# %%
# Define a basic Neural Network

class ComplexRegressor(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ComplexRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# %%
### CPU: 8m3s, Best Score (RMSLE): tensor(0.1523)
### GPU: 1m48s, Best Score (RMSLE): tensor(0.1537)
### Cyclic LR GPU (Adam): 2m20s, Best Score (RMSLE): tensor(0.1519)
### Cyclic LR GPU (Adamax): 2m33s, Best Score (RMSLE): tensor(0.1529)
### Cyclic LR GPU (SGD): 1m59s, Best Score (RMSLE): tensor(0.1523

# %%
### CPU no kfold cv
### CPU core usage around 60%
### Temps holding around 65 celsius 

best_score = None
best_params = None

# Define your params for different neural network architectures
params = {
    'architectures': [ComplexRegressor],
    'hidden_sizes': [1024],
    'learning_rates': [0.01],
    'num_epochs': 100,
    'batch_size': 64,
}


for hidden_size1 in params['hidden_sizes']:
    for hidden_size2 in params['hidden_sizes']:
        for learning_rate in params['learning_rates']:
            # Initialize model, loss function, optimizer
            model = ComplexRegressor(input_size=X_train_cpu.shape[1], hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=1)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Create DataLoader with the specified batch size
            train_dataset = TensorDataset(X_train_cpu, y_train_cpu)
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

            # Training loop
            for epoch in range(params['num_epochs']):
                model.train()
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    outputs = outputs.squeeze(dim=1)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_cpu)
                val_outputs = val_outputs.squeeze(dim=1)
                val_loss = criterion(val_outputs, y_test_cpu)
                val_loss = val_loss.item()

            # Calculate RMSLE or any other metric
            score = rmsle(val_outputs, y_test_cpu)

            # Update best_score and best_params based on score
            if best_score is None or score < best_score:
                best_score = score
                best_params = {
                    'hidden_size1': hidden_size1,
                    'hidden_size2': hidden_size2,
                    'learning_rate': learning_rate,
                    'score': score
                }


print("Best Parameters:", best_params)
print("Best Score (RMSLE):", best_score)

# %%
### CPU with 5 fold cv
### CPU core usage arund 55%
### CPU temp around 65c

best_score = None
best_params = None

# Define your params for different neural network architectures
params = {
    'architectures': [ComplexRegressor],
    'hidden_sizes': [1024],
    'learning_rates': [0.01],
    'num_epochs': 100,
    'batch_size': 64,
}

# Iterate over each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X_cpu)):
    print(f'Fold {fold + 1}/{kf.n_splits}')

    # Split the data for this fold and convert to PyTorch tensors
    X_train_fold = torch.tensor(X_cpu[train_idx], dtype=torch.float32)
    y_train_fold = torch.tensor(y_cpu[train_idx], dtype=torch.float32)
    X_val_fold = torch.tensor(X_cpu[val_idx], dtype=torch.float32)
    y_val_fold = torch.tensor(y_cpu[val_idx], dtype=torch.float32)

    for hidden_size1 in params['hidden_sizes']:
        for hidden_size2 in params['hidden_sizes']:
            for learning_rate in params['learning_rates']:
                # Initialize model, loss function, optimizer
                model = ComplexRegressor(input_size=X_train_fold.shape[1], hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=1)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                # Create DataLoader with the specified batch size
                train_dataset = TensorDataset(X_train_fold, y_train_fold)
                train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

                # Training loop
                for epoch in range(params['num_epochs']):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        outputs = outputs.squeeze(dim=1)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()

                # Evaluation on the validation fold
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_fold)
                    val_outputs = val_outputs.squeeze(dim=1)
                    val_loss = criterion(val_outputs, y_val_fold)
                    val_loss = val_loss.item()

                # Calculate RMSLE or any other metric
                score = rmsle(val_outputs, y_val_fold)

                # Update best_score and best_params based on score
                if best_score is None or score < best_score:
                    best_score = score
                    best_params = {
                        'hidden_size1': hidden_size1,
                        'hidden_size2': hidden_size2,
                        'learning_rate': learning_rate,
                        'score': score
                    }
                
                fold_scores.append(score)
                print(f'Fold {fold + 1} Score: {score:.4f}')

# Average the scores across all folds
average_score = np.mean(fold_scores)
print(f'Average K-Fold Validation Score (RMSLE): {average_score:.4f}')
print("Best Parameters:", best_params)

# %%
### GPU accelerated prototyping 
### this is done by setting model=... .to(device)
### criterion=... .to(device)

### GPU core load aroun 40%
### GPU temp around 55 degrees celsius

best_score = None
best_params = None

# Define your params for different neural network architectures
params = {
    'architectures': [ComplexRegressor],
    'hidden_sizes': [1024],
    'learning_rates': [0.01],
    'num_epochs': 100,
    'batch_size': 64,
}

for hidden_size1 in params['hidden_sizes']:
    for hidden_size2 in params['hidden_sizes']:
        for learning_rate in params['learning_rates']:
            # Initialize model, loss function, optimizer and move them to GPU
            model = ComplexRegressor(input_size=X_train.shape[1], hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=1).to(device)
            criterion = nn.MSELoss().to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Create DataLoader with the specified batch size
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

            # Training loop
            for epoch in range(params['num_epochs']):
                model.train()
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    outputs = outputs.squeeze(dim=1)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                val_outputs = val_outputs.squeeze(dim=1)
                val_loss = criterion(val_outputs, y_test)
                val_loss = val_loss.item()

            # Calculate RMSLE or any other metric
            score = rmsle(val_outputs, y_test)

            # Update best_score and best_params based on score
            if best_score is None or score < best_score:
                best_score = score
                best_params = {
                    'hidden_size1': hidden_size1,
                    'hidden_size2': hidden_size2,
                    'learning_rate': learning_rate,
                    'score': score
                }
                
print("Best Parameters:", best_params)
print("Best Score (RMSLE):", best_score)

# %%
### GPU 5 fold cv
### this is done by setting model=... .to(device)
### criterion=... .to(device)

### GPU core load aroun 40%
### GPU temp around 55 degrees celsius

kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_score = None
best_params = None

# Cross-validation loop
for hidden_size1 in params['hidden_sizes']:
    for hidden_size2 in params['hidden_sizes']:
        for learning_rate in params['learning_rates']:
            fold_scores = []

            for train_index, val_index in kf.split(X_gpu):
                X_train_fold, X_val_fold = X_gpu[train_index], X_gpu[val_index]
                y_train_fold, y_val_fold = y_gpu[train_index], y_gpu[val_index]

                # Initialize model, loss function, and optimizer
                model = ComplexRegressor(input_size=X_train_fold.shape[1], hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=1).to(device)
                criterion = nn.MSELoss().to(device)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                # Create DataLoader with the specified batch size
                train_dataset = TensorDataset(X_train_fold, y_train_fold)
                train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

                # Training loop
                for epoch in range(params['num_epochs']):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # Move to GPU
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        outputs = outputs.squeeze(dim=1)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()

                # Evaluation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_fold)
                    val_outputs = val_outputs.squeeze(dim=1)
                    val_loss = criterion(val_outputs, y_val_fold)
                    val_loss = val_loss.item()

                # Calculate RMSLE or any other metric
                score = rmsle(val_outputs, y_val_fold)

                fold_scores.append(score)
                
                fold_scores_cpu = [score.cpu().numpy() for score in fold_scores]

            # Average score for current parameters
            avg_score = np.mean(fold_scores_cpu)

            # Update best_score and best_params based on avg_score
            if best_score is None or avg_score < best_score:
                best_score = avg_score
                best_params = {
                    'hidden_size1': hidden_size1,
                    'hidden_size2': hidden_size2,
                    'learning_rate': learning_rate,
                    'score': best_score
                }

print("Best Parameters:", best_params)
print("Best Score (RMSLE):", best_score)


# %%
### Cyclic learning rate with Adam
### GPU core load aroun 40%
### GPU temp around 55 degrees celsius

# Define your parameters for different neural network architectures
params = {
    'architectures': [ComplexRegressor],
    'hidden_sizes': [1024],
    'learning_rates': [0.01],
    'num_epochs': 100,
    'batch_size': 64,
    'base_lr': 1e-8,  # Base learning rate for cyclic learning rate scheduler
    'max_lr': 1e-2,   # Maximum learning rate for cyclic learning rate scheduler
    'step_size_up': 3000  # Number of iterations for the increasing phase
}

# Train and evaluate different parameters. For this I only used one value for 'hidden_sizes', 'learning_rates', and 'architectures'
best_score = None
best_params = None


for hidden_size1 in params['hidden_sizes']:
    for hidden_size2 in params['hidden_sizes']:
        for learning_rate in params['learning_rates']:
            # Initialize model, loss function, optimizer and scheduler
            model = ComplexRegressor(input_size=X_train.shape[1], hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=1).to(device)
            criterion = nn.MSELoss().to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = CyclicLR(
                optimizer,
                base_lr=params['base_lr'],
                max_lr=params['max_lr'],
                step_size_up=params['step_size_up'],
                mode='triangular',  # or 'triangular2' or 'exp_range'
                cycle_momentum=False  # Adam optimizer does not use momentum
            )

            # Create DataLoader with the specified batch size
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

            # Training loop
            for epoch in range(params['num_epochs']):
                model.train()
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    outputs = outputs.squeeze(dim=1)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()  # Update the learning rate based on the scheduler
                    epoch_loss += loss.item()

                # avg_loss = epoch_loss / len(train_loader)
                # current_lr = scheduler.get_last_lr()[0]
                # print(f"Epoch {epoch+1}/{params['num_epochs']}, Loss: {avg_loss:.4f}, Learning Rate: {current_lr:.6f}")

            # Evaluation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                val_outputs = val_outputs.squeeze(dim=1)
                val_loss = criterion(val_outputs, y_test).item()

            # Calculate RMSLE or any other metric
            score = rmsle(val_outputs, y_test)

            # Update best_score and best_params based on score
            if best_score is None or score < best_score:
                best_score = score
                best_params = {
                    'hidden_size1': hidden_size1,
                    'hidden_size2': hidden_size2,
                    'learning_rate': learning_rate,
                    'score': score
                    }

print("Best Parameters:", best_params)
print("Best Score (RMSLE):", best_score)

# %%
### Cyclic learning rate with Adam and 5 fold cv
### GPU core load aroun 40%
### GPU temp around 55 degrees celsius

# Define your parameters for different neural network architectures
params = {
    'architectures': [ComplexRegressor],
    'hidden_sizes': [1024],
    'learning_rates': [0.01],
    'num_epochs': 100,
    'batch_size': 64,
    'base_lr': 1e-8,  # Base learning rate for cyclic learning rate scheduler
    'max_lr': 1e-2,   # Maximum learning rate for cyclic learning rate scheduler
    'step_size_up': 3000,  # Number of iterations for the increasing phase
    'k_folds': 5
}

kf = KFold(n_splits=params['k_folds'], shuffle=True, random_state=42)

# Train and evaluate different parameters. For this I only used one value for 'hidden_sizes', 'learning_rates', and 'architectures'
best_score = None
best_params = None


# Cross-validation loop
for hidden_size1 in params['hidden_sizes']:
    for hidden_size2 in params['hidden_sizes']:
        for learning_rate in params['learning_rates']:
            fold_scores = []

            for train_index, val_index in kf.split(X_gpu):
                X_train_fold, X_val_fold = X_gpu[train_index], X_gpu[val_index]
                y_train_fold, y_val_fold = y_gpu[train_index], y_gpu[val_index]

                # Initialize model, loss function, optimizer, and scheduler
                model = ComplexRegressor(input_size=X_train_fold.shape[1], hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=1).to(device)
                criterion = nn.MSELoss().to(device)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                scheduler = CyclicLR(
                    optimizer,
                    base_lr=params['base_lr'],
                    max_lr=params['max_lr'],
                    step_size_up=params['step_size_up'],
                    mode='triangular',  # or 'triangular2' or 'exp_range'
                    cycle_momentum=False  # Adam optimizer does not use momentum
                )

                # Create DataLoader with the specified batch size
                train_dataset = TensorDataset(X_train_fold, y_train_fold)
                train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

                # Training loop
                for epoch in range(params['num_epochs']):
                    model.train()
                    epoch_loss = 0
                    for batch_X, batch_y in train_loader:
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        outputs = outputs.squeeze(dim=1)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        scheduler.step()  # Update the learning rate based on the scheduler
                        epoch_loss += loss.item()

                # Evaluation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_fold.to(device))
                    val_outputs = val_outputs.squeeze(dim=1)
                    val_loss = criterion(val_outputs, y_val_fold.to(device)).item()

                # Calculate RMSLE or any other metric
                score = rmsle(val_outputs.cpu(), y_val_fold.cpu())  # Ensure CPU for metric calculation

                fold_scores.append(score)
                
                fold_scores_cpu = [score.cpu().numpy() for score in fold_scores]

            # Average score for current parameters
            avg_score = np.mean(fold_scores_cpu)

            # Update best_score and best_params based on avg_score
            if best_score is None or avg_score < best_score:
                best_score = avg_score
                best_params = {
                    'hidden_size1': hidden_size1,
                    'hidden_size2': hidden_size2,
                    'learning_rate': learning_rate,
                    'score': best_score
                }

print("Best Parameters:", best_params)
print("Best Score (RMSLE):", best_score)

# %%
### Cyclic learning rate with Adamax
### GPU core load aroun 40%
### GPU temp around 55 degrees celsius

# Define your parameters for different neural network architectures
params = {
    'architectures': [ComplexRegressor],
    'hidden_sizes': [1024],
    'learning_rates': [0.01],
    'num_epochs': 100,
    'batch_size': 64,
    'base_lr': 1e-8,  # Base learning rate for cyclic learning rate scheduler
    'max_lr': 1e-2,   # Maximum learning rate for cyclic learning rate scheduler
    'step_size_up': 3000  # Number of iterations for the increasing phase
}

# Train and evaluate different parameters. For this I only used one value for 'hidden_sizes', 'learning_rates', and 'architectures'
best_score = None
best_params = None


for hidden_size1 in params['hidden_sizes']:
    for hidden_size2 in params['hidden_sizes']:
        for learning_rate in params['learning_rates']:
            # Initialize model, loss function, optimizer and scheduler
            model = ComplexRegressor(input_size=X_train.shape[1], hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=1).to(device)
            criterion = nn.MSELoss().to(device)
            optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
            scheduler = CyclicLR(
                optimizer,
                base_lr=params['base_lr'],
                max_lr=params['max_lr'],
                step_size_up=params['step_size_up'],
                mode='triangular',  # or 'triangular2' or 'exp_range'
                cycle_momentum=False  # Adam optimizer does not use momentum
            )

            # Create DataLoader with the specified batch size
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

            # Training loop
            for epoch in range(params['num_epochs']):
                model.train()
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    outputs = outputs.squeeze(dim=1)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()  # Update the learning rate based on the scheduler
                    epoch_loss += loss.item()

                # avg_loss = epoch_loss / len(train_loader)
                # current_lr = scheduler.get_last_lr()[0]
                # print(f"Epoch {epoch+1}/{params['num_epochs']}, Loss: {avg_loss:.4f}, Learning Rate: {current_lr:.6f}")

            # Evaluation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                val_outputs = val_outputs.squeeze(dim=1)
                val_loss = criterion(val_outputs, y_test).item()

            # Calculate RMSLE or any other metric
            score = rmsle(val_outputs, y_test)

            # Update best_score and best_params based on score
            if best_score is None or score < best_score:
                best_score = score
                best_params = {
                    'hidden_size1': hidden_size1,
                    'hidden_size2': hidden_size2,
                    'learning_rate': learning_rate,
                    'score': score
                    }

print("Best Parameters:", best_params)
print("Best Score (RMSLE):", best_score)

# %%
### Cyclic learning rate with Adamax and 5 fold cv
### GPU core load aroun 40%
### GPU temp around 55 degrees celsius

# Define your parameters for different neural network architectures
params = {
    'architectures': [ComplexRegressor],
    'hidden_sizes': [1024],
    'learning_rates': [0.01],
    'num_epochs': 100,
    'batch_size': 64,
    'base_lr': 1e-8,  # Base learning rate for cyclic learning rate scheduler
    'max_lr': 1e-2,   # Maximum learning rate for cyclic learning rate scheduler
    'step_size_up': 3000,  # Number of iterations for the increasing phase
    'k_folds': 5
}

kf = KFold(n_splits=params['k_folds'], shuffle=True, random_state=42)

# Train and evaluate different parameters. For this I only used one value for 'hidden_sizes', 'learning_rates', and 'architectures'
best_score = None
best_params = None


# Cross-validation loop
for hidden_size1 in params['hidden_sizes']:
    for hidden_size2 in params['hidden_sizes']:
        for learning_rate in params['learning_rates']:
            fold_scores = []

            for train_index, val_index in kf.split(X):
                X_train_fold, X_val_fold = X[train_index], X[val_index]
                y_train_fold, y_val_fold = y[train_index], y[val_index]

                # Initialize model, loss function, optimizer, and scheduler
                model = ComplexRegressor(input_size=X_train_fold.shape[1], hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=1).to(device)
                criterion = nn.MSELoss().to(device)
                optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
                scheduler = CyclicLR(
                    optimizer,
                    base_lr=params['base_lr'],
                    max_lr=params['max_lr'],
                    step_size_up=params['step_size_up'],
                    mode='triangular',  # or 'triangular2' or 'exp_range'
                    cycle_momentum=False  # Adam optimizer does not use momentum
                )

                # Create DataLoader with the specified batch size
                train_dataset = TensorDataset(X_train_fold, y_train_fold)
                train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

                # Training loop
                for epoch in range(params['num_epochs']):
                    model.train()
                    epoch_loss = 0
                    for batch_X, batch_y in train_loader:
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        outputs = outputs.squeeze(dim=1)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        scheduler.step()  # Update the learning rate based on the scheduler
                        epoch_loss += loss.item()

                # Evaluation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_fold.to(device))
                    val_outputs = val_outputs.squeeze(dim=1)
                    val_loss = criterion(val_outputs, y_val_fold.to(device)).item()

                # Calculate RMSLE or any other metric
                score = rmsle(val_outputs.cpu(), y_val_fold.cpu())  # Ensure CPU for metric calculation

                fold_scores.append(score)

            # Average score for current parameters
            avg_score = np.mean(fold_scores)

            # Update best_score and best_params based on avg_score
            if best_score is None or avg_score < best_score:
                best_score = avg_score
                best_params = {
                    'hidden_size1': hidden_size1,
                    'hidden_size2': hidden_size2,
                    'learning_rate': learning_rate,
                    'score': best_score
                }

print("Best Parameters:", best_params)
print("Best Score (RMSLE):", best_score)

# %%
### Cyclic learning rate with SGD
### Momentum = 0.9
### GPU core load aroun 40%
### GPU temp around 55 degrees celsius

# Define your parameters for different neural network architectures
params = {
    'architectures': [ComplexRegressor],
    'hidden_sizes': [1024],
    'learning_rates': [0.01],
    'momentum': [.9],
    'num_epochs': 100,
    'batch_size': 64,
    'base_lr': 1e-8,  # Base learning rate for cyclic learning rate scheduler
    'max_lr': 1e-2,   # Maximum learning rate for cyclic learning rate scheduler
    'step_size_up': 3000  # Number of iterations for the increasing phase
}

# Train and evaluate different parameters. For this I only used one value for 'hidden_sizes', 'learning_rates', and 'architectures'
best_score = None
best_params = None


for hidden_size1 in params['hidden_sizes']:
    for hidden_size2 in params['hidden_sizes']:
        for learning_rate in params['learning_rates']:
            # Initialize model, loss function, optimizer and scheduler
            model = ComplexRegressor(input_size=X_train.shape[1], hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=1).to(device)
            criterion = nn.MSELoss().to(device)
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=.9)
            scheduler = CyclicLR(
                optimizer,
                base_lr=params['base_lr'],
                max_lr=params['max_lr'],
                step_size_up=params['step_size_up'],
                mode='triangular',  # or 'triangular2' or 'exp_range'
                # cycle_momentum=False  # Adam optimizer does not use momentum
            )

            # Create DataLoader with the specified batch size
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

            # Training loop
            for epoch in range(params['num_epochs']):
                model.train()
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    outputs = outputs.squeeze(dim=1)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()  # Update the learning rate based on the scheduler
                    epoch_loss += loss.item()

                # avg_loss = epoch_loss / len(train_loader)
                # current_lr = scheduler.get_last_lr()[0]
                # print(f"Epoch {epoch+1}/{params['num_epochs']}, Loss: {avg_loss:.4f}, Learning Rate: {current_lr:.6f}")

            # Evaluation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                val_outputs = val_outputs.squeeze(dim=1)
                val_loss = criterion(val_outputs, y_test).item()

            # Calculate RMSLE or any other metric
            score = rmsle(val_outputs, y_test)

            # Update best_score and best_params based on score
            if best_score is None or score < best_score:
                best_score = score
                best_params = {
                    'hidden_size1': hidden_size1,
                    'hidden_size2': hidden_size2,
                    'learning_rate': learning_rate,
                    'score': score
                    }

print("Best Parameters:", best_params)
print("Best Score (RMSLE):", best_score)

# %%
### Cyclic learning rate with SGD and 5 fold cv
### GPU core load aroun 40%
### GPU temp around 55 degrees celsius

# Define your parameters for different neural network architectures
params = {
    'architectures': [ComplexRegressor],
    'hidden_sizes': [1024],
    'learning_rates': [0.01],
    'num_epochs': 100,
    'batch_size': 64,
    'base_lr': 1e-8,  # Base learning rate for cyclic learning rate scheduler
    'max_lr': 1e-2,   # Maximum learning rate for cyclic learning rate scheduler
    'step_size_up': 3000,  # Number of iterations for the increasing phase
    'k_folds': 5
}

kf = KFold(n_splits=params['k_folds'], shuffle=True, random_state=42)

# Train and evaluate different parameters. For this I only used one value for 'hidden_sizes', 'learning_rates', and 'architectures'
best_score = None
best_params = None


# Cross-validation loop
for hidden_size1 in params['hidden_sizes']:
    for hidden_size2 in params['hidden_sizes']:
        for learning_rate in params['learning_rates']:
            fold_scores = []

            for train_index, val_index in kf.split(X):
                X_train_fold, X_val_fold = X[train_index], X[val_index]
                y_train_fold, y_val_fold = y[train_index], y[val_index]

                # Initialize model, loss function, optimizer, and scheduler
                model = ComplexRegressor(input_size=X_train_fold.shape[1], hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=1).to(device)
                criterion = nn.MSELoss().to(device)
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=.9)
                scheduler = CyclicLR(
                    optimizer,
                    base_lr=params['base_lr'],
                    max_lr=params['max_lr'],
                    step_size_up=params['step_size_up'],
                    mode='triangular',  # or 'triangular2' or 'exp_range'
                    cycle_momentum=False  # Adam optimizer does not use momentum
                )

                # Create DataLoader with the specified batch size
                train_dataset = TensorDataset(X_train_fold, y_train_fold)
                train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

                # Training loop
                for epoch in range(params['num_epochs']):
                    model.train()
                    epoch_loss = 0
                    for batch_X, batch_y in train_loader:
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        outputs = outputs.squeeze(dim=1)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        scheduler.step()  # Update the learning rate based on the scheduler
                        epoch_loss += loss.item()

                # Evaluation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_fold.to(device))
                    val_outputs = val_outputs.squeeze(dim=1)
                    val_loss = criterion(val_outputs, y_val_fold.to(device)).item()

                # Calculate RMSLE or any other metric
                score = rmsle(val_outputs.cpu(), y_val_fold.cpu())  # Ensure CPU for metric calculation

                fold_scores.append(score)

            # Average score for current parameters
            avg_score = np.mean(fold_scores)

            # Update best_score and best_params based on avg_score
            if best_score is None or avg_score < best_score:
                best_score = avg_score
                best_params = {
                    'hidden_size1': hidden_size1,
                    'hidden_size2': hidden_size2,
                    'learning_rate': learning_rate,
                    'score': best_score
                }

print("Best Parameters:", best_params)
print("Best Score (RMSLE):", best_score)

# %%


# %%