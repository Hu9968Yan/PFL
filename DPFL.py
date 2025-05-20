from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
import copy
import os
import timm

seed = 42
global_epoch = 100
batchsize = 16

# 创建保存模型的目录
save_dir = r'D:\HuYan\Data_and_model\BT\savemodel\validate520\densenet121'
os.makedirs(save_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Data preprocessing and augmentation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

client_dataset = []
for root in [r'D:\HuYan\Data_and_model\BT\client1',
             r'D:\HuYan\Data_and_model\BT\client2',
             r'D:\HuYan\Data_and_model\BT\client3',
             r'D:\HuYan\Data_and_model\BT\client4']:
    client_dataset.append(ImageFolder(root, transform=data_transforms))
loader3 = []
for d in client_dataset:
    n_train = int(len(d) * 0.7)
    n_val = int(len(d) * 0.1)
    n_test = len(d) - n_train - n_val
    train_set, val_set, test_set = random_split(d, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed))
# 随机分割数据集，固定随机种子确保结果可复现
    train_loader = DataLoader(train_set, batchsize, True)
    val_loader = DataLoader(val_set, batchsize, False)
    test_loader = DataLoader(test_set, batchsize, False)
    loader3.append([train_loader, val_loader, test_loader])

class TumorClassifier(nn.Module):
    def __init__(self):
        super(TumorClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# global_model = TumorClassifier()
global_model = timm.create_model('densenet121', pretrained=False, num_classes=4)
# global_model = timm.create_model('resnet50', pretrained=False, num_classes=4)
# global_model = timm.create_model('mobilenetv2_100', pretrained=False, num_classes=4)
# global_model = timm.create_model('convnext_base', pretrained=False, num_classes=4)
# global_model = timm.create_model('edgenext_base.in21k_ft_in1k', pretrained=False, num_classes=4)
mu = 0.01  # 一个正则化参数，用于控制客户端模型与全局模型之间的权重差异对损失的影响

class Client(object):
    def __init__(self, dataloader, learning_rate, epochs, idx):
        self.train_loader = dataloader
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = global_model
        self.idx = idx
        self.global_step = 0
        self.train_losses = []
        self.train_accuracies = []
        self.lastmodel = global_model
        self.lastacc = 0
        self.val_losses = []
        self.val_accuracies = []
        self.bestacc = 0
        self.best_model = global_model
        self.val_loader = loader3[idx - 1][1]
        self.test_loader = loader3[idx - 1][2]
        self.fusion_model = global_model

    def train(self):
        print("------" + str(self.idx) + " strat-------")
        self.model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        e_loss = []
        for epoch in range(1, self.epochs + 1):

            train_loss = 0.0
            correct = 0
            total = 0

            self.model.train()
            for batch_idx, (data, labels) in enumerate(self.train_loader):
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                output = self.model(data)

                pt = 0
                for w, w_t in zip(self.model.parameters(), global_model.parameters()):
                    pt = (w - w_t).pow(2).sum().item() + pt
                loss = criterion(output, labels) + (mu / 2) * pt
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = train_loss / len(self.train_loader.dataset)
            train_accuracy = correct / total
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)

        train_loss /= len(self.train_loader)
        return self.model.state_dict(), train_loss

    def upgrade(self, accweight):
        dict1 = global_model.state_dict()#获取模型参数
        dict2 = self.model.state_dict()
        for k, v in dict1.items():
            dict1[k] = dict1[k] * (1 - (0.7 - (0.25 - accweight[self.idx - 1]))) + dict2[k] * (
                    0.7 - (0.25 - accweight[self.idx - 1]))
        self.fusion_model.load_state_dict(dict1)
        g_val_loss, g_val_accuracy = self.validate2(global_model)
        last_val_loss, last_val_accuracy = self.validate2(self.lastmodel)
        local_val_loss, local_val_accuracy = self.validate2(self.model)
        fusion_val_loss, fusion_val_accuracy = self.validate2(self.fusion_model)

        acclist2 = [g_val_accuracy, last_val_accuracy, local_val_accuracy, fusion_val_accuracy]
        if max(acclist2) == g_val_accuracy:
            self.model = global_model
        elif max(acclist2) == last_val_accuracy:
            self.model = self.lastmodel
        elif max(acclist2) == local_val_accuracy:
            self.model = self.model
        elif max(acclist2) == fusion_val_accuracy:
            self.model = self.fusion_model

        self.lastmodel = self.model
        self.global_step += 1

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(self.val_loader)
        val_accuracy = correct / total
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)
        step = self.global_step
        self.lastmodel = self.model
        self.lastacc = val_accuracy

        self.test(self.lastmodel)

        print("Epoch: " + str(step + 1))
        print("Idx: " + str(self.idx))
        print("Training Loss: " + str(self.train_losses[-1]))
        print("Training Accuracy: " + str(self.train_accuracies[-1]))
        print("Validation Loss: " + str(val_loss))
        print("Validation Accuracy: " + str(val_accuracy))

        if val_accuracy > self.bestacc:
            self.bestacc = val_accuracy
            self.best_model = copy.deepcopy(self.model)

        return val_accuracy

    def validate2(self, model):
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(self.val_loader)
        val_accuracy = correct / total

        return val_loss, val_accuracy

    def test(self, model):
        test_loader = self.test_loader
        target_num = torch.zeros((1, 4))
        predict_num = torch.zeros((1, 4))
        acc_num = torch.zeros((1, 4))
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pre_mask = torch.zeros(outputs.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
                predict_num += pre_mask.sum(0)
                tar_mask = torch.zeros(outputs.size()).scatter_(1, labels.data.cpu().view(-1, 1), 1.)
                target_num += tar_mask.sum(0)
                acc_mask = pre_mask * tar_mask
                acc_num += acc_mask.sum(0)

        recall = acc_num / target_num
        precision = acc_num / predict_num
        F1 = 2 * recall * precision / (recall + precision)
        test_accuracy = acc_num.sum(1) / target_num.sum(1)

        test_loss /= len(test_loader)

        if test_accuracy > self.bestacc:
            self.bestacc = test_accuracy
            self.recall = recall
            self.precision = precision
            self.F1 = F1

    def log(self):
        print("Final_Best:")
        print(self.idx)
        print(self.bestacc)
        print(self.recall)
        print(self.precision)
        print(self.F1)
        save_path = os.path.join(save_dir, f'client{self.idx}_best_model.pth')
        torch.save(self.best_model.state_dict(), save_path)
        print(f"Best model of client {self.idx} saved to {save_path}")


client1 = Client(loader3[0][0], 0.001, 2, 1)
client2 = Client(loader3[1][0], 0.001, 2, 2)
client3 = Client(loader3[2][0], 0.001, 2, 3)
client4 = Client(loader3[3][0], 0.001, 2, 4)
Client_all = [client1, client2, client3, client4]

train_loss = []
for i in range(global_epoch):
    w, local_loss = [], []
    acclist = []
    for c in Client_all:
        weights, loss = c.train()
        w.append(copy.deepcopy(weights))
        local_loss.append(copy.deepcopy(loss))
        valacc = c.validate()
        acclist.append(valacc)

    totalacc = sum(acclist)
    accweight = [acc / totalacc for acc in acclist]
    dataweight = [0.1, 0.2, 0.3, 0.4]
    print(accweight)

    weights_avg = copy.deepcopy(w[0])
    for k in weights_avg.keys():
        for i in range(1, len(w)):
            weights_avg[k] += w[i][k]
        weights_avg[k] = torch.div(weights_avg[k], len(w))
    global_weights = weights_avg
    global_model.load_state_dict(global_weights)

    for c in Client_all:
        c.upgrade(accweight)
        c.epochs = 2

    minindex = accweight.index(min(accweight))
    Client_all[minindex].epochs = 2

for c in Client_all:
    c.log()