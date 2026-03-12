import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import argparse
from MovieNet import MovieNet

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fonction pour entraîner le modèle
def train(net, optimizer, loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)
        for x, y in t:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
            t.set_description(f'Epoch {epoch+1}, training loss: {sum(running_loss)/len(running_loss):.4f}')
        print(f"Epoch {epoch+1} loss: {sum(running_loss)/len(running_loss):.4f}")

# Fonction pour tester le modèle
def test(net, loader):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_hat = net(x).argmax(1)
            correct += (y_hat == y).sum().item()
            total += y.size(0)
    net.train()
    return correct / total

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    # Transforms pour les posters
    transform = transforms.Compose([
        transforms.Resize((185, 298)),  # garder la taille d'origine ou ajuster
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    
    dataset = torchvision.datasets.ImageFolder(
        '/home/senatorequentin/MovieGenre/content/sorted_movie_posters_paligema',  # A modifier en fonction de où est situé le dataset !!!
        transform=transform
    )
    
    # Split train/test (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Instanciation du modèle
    net = MovieNet()  # remplacer par MoviePosterNet adapté à 3 canaux et 10 genres
    net = net.to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    
    # Training
    train(net, optimizer, trainloader, epochs=args.epochs)
    
    # Testing
    acc = test(net, testloader)
    print(f'Test accuracy: {acc:.4f}')
    
    # Création du dossier des poids s'il n'existe pas
    os.makedirs('saved_models', exist_ok=True)

    # Sauvegarde des poids
    torch.save(net.state_dict(), 'saved_models/movie_poster_model.pth', _use_new_zipfile_serialization=False)