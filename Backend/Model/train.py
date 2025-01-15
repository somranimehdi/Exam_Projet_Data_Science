import torch
from torch.utils.data import Subset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import numpy as np

def train(model, train_loader, test_loader, criterion, optimizer, device, epochs=10):
    model.to(device)
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {running_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        print(f"Validation Loss: {val_loss:.4f}")

    return train_losses, val_losses



def cross_validate(model, dataset, criterion, optimizer, device, epochs=15, k_folds=5, batch_size=64):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    results = {
        'fold': [],
        'train_losses': [],
        'val_losses': [],
    }
    for fold, (train_indices, val_indices) in enumerate(kfold.split(np.arange(len(dataset)))):
        print(f'FOLD {fold + 1}/{k_folds}')
        print('--------------------------------')
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        model.apply(reset_weights)
        
        train_losses, val_losses = train(model, train_loader, val_loader, criterion, optimizer, device, epochs=epochs)
        
        results['fold'].append(fold + 1)
        results['train_losses'].append(train_losses)
        results['val_losses'].append(val_losses)
    
    return results

def reset_weights(layer):
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
##example of usage # Perform cross-validation
#results = cross_validate(model, dataset, criterion, optimizer, device, epochs=5, k_folds=5, batch_size=64)

## Print results
#for fold, train_loss, val_loss in zip(results['fold'], results['train_losses'], results['val_losses']):
   # print(f"Fold {fold}: Train Loss = {np.mean(train_loss):.4f}, Val Loss = {np.mean(val_loss):.4f}")