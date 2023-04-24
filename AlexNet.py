import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim

import os
import json
from tqdm import tqdm

from PIL import Image

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # input[3,224,224] output[96,55,55]
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(True),
            # output[96,27,27]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # output[256,27,27]
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # output[256,13,13]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # output[384,13,13]
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[384,13,13]
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[256,13,13]
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[256,6,6]
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(0.5),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"using {device} device")

    img_path = os.path.join(os.getcwd(), "../data/flower_data/")
    assert os.path.exists(img_path), print(
        f"img_path {img_path} does not exist.")

    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        "validation": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    }
    train_dataset = datasets.ImageFolder(root=os.path.join(
        img_path, "train"), transform=data_transform["train"])
    validation_dataset = datasets.ImageFolder(root=os.path.join(
        img_path, "validation"), transform=data_transform["validation"])
    train_num = len(train_dataset)
    validation_num = len(validation_dataset)
    print(
        f"using {train_num} images for train, using {validation_num} images for validation.")

    batch_size = 32
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    validation_loader = DataLoader(
        validation_dataset, batch_size=4, shuffle=False, num_workers=10)

    model = AlexNet(5)
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    epochs = 10
    save_path = './AlexNet.pth'
    best_acc = 0.0
    for epoch in range(epochs):
        train_acc = 0.0
        train_loss = 0.0
        validation_acc = 0.0
        validation_loss = 0.0
        #
        model.train()
        for i, batch in enumerate(tqdm(train_loader)):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            _, train_pred = torch.max(outputs, 1)
            train_acc += (train_pred.detach() == labels.detach()).sum().item()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(validation_loader)):
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)

                loss = loss_function(outputs, labels)

                _, validation_pred = torch.max(outputs, 1)
                validation_acc += (validation_pred.cpu() == labels.cpu()).sum().item()
                validation_loss += loss.item()
        print(f'[{epoch+1:03d}/{epochs:03d}] Train Acc: {train_acc/train_num:3.5f} '
              f'Loss: {train_loss/len(train_loader):3.5f} | Validation Acc: {validation_acc/validation_num:3.5f} '
              f'Loss: {validation_loss/len(validation_loader):3.5f}')
        if validation_acc > best_acc:
            best_acc = validation_acc
            torch.save(model.state_dict(), save_path)
            print(f'saving model with acc {best_acc/validation_num:.5f}')

def predict():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device")
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    img_path = '../data/2.jpg'
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    model = AlexNet(5).to(device)
    weights_path = "./AlexNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    with torch.no_grad():  
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    print(f"{predict_cla}, {predict}")
    

if __name__ == '__main__':
    main()
    predict()
