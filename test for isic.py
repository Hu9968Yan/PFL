import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import timm

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置随机种子
random_seed = 42
torch.manual_seed(random_seed)

# 数据预处理和增强
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 自定义数据集划分函数
def split_dataset(dataset, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    assert train_ratio + val_ratio + test_ratio == 1.0

    # 先划分训练+验证和测试集
    trainval_size = int(len(dataset) * (train_ratio + val_ratio))
    test_size = len(dataset) - trainval_size
    trainval_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [trainval_size, test_size], generator=torch.Generator().manual_seed(random_seed))

    # 再划分训练和验证集
    train_size = int(trainval_size * (train_ratio / (train_ratio + val_ratio)))
    val_size = trainval_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        trainval_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(random_seed))

    return train_dataset, val_dataset, test_dataset


# 初始化各客户端数据集
clients = []
# client_paths = [
#     r'D:\HuYan\Chen-Xianrui\dirichlet\datasets\BT\alpha_100\client1',
#     r'D:\HuYan\Chen-Xianrui\dirichlet\datasets\BT\alpha_100\client2',
#     r'D:\HuYan\Chen-Xianrui\dirichlet\datasets\BT\alpha_100\client3',
#     r'D:\HuYan\Chen-Xianrui\dirichlet\datasets\BT\alpha_100\client4'
# ]

client_paths = [r'D:\HuYan\BT\client1',
                r'D:\HuYan\BT\client2',
                r'D:\HuYan\BT\client3',
                r'D:\HuYan\BT\client4']
for path in client_paths:
    full_dataset = ImageFolder(path, transform=data_transforms)
    train_set, val_set, test_set = split_dataset(full_dataset)
    clients.append({
        'train': DataLoader(train_set, batch_size=16, shuffle=True),
        'val': DataLoader(val_set, batch_size=16, shuffle=False),
        'test': DataLoader(test_set, batch_size=16, shuffle=False)
    })


# 测试函数
def test(model, test_loader):
    model.eval()
    target_num = torch.zeros((1, 4))  # 4 是 class_num
    predict_num = torch.zeros((1, 4))
    acc_num = torch.zeros((1, 4))
    test_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
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

    # 处理除以零的情况
    valid_target_mask = target_num > 0
    valid_predict_mask = predict_num > 0
    recall = torch.where(valid_target_mask, acc_num / target_num, torch.tensor(0.0))
    precision = torch.where(valid_predict_mask, acc_num / predict_num, torch.tensor(0.0))

    # 避免 F1 计算时出现除以零
    f1_numerator = 2 * recall * precision
    f1_denominator = recall + precision
    valid_f1_mask = f1_denominator > 0
    F1 = torch.where(valid_f1_mask, f1_numerator / f1_denominator, torch.tensor(0.0))

    test_accuracy = acc_num.sum(1) / target_num.sum(1)
    test_loss /= len(test_loader)
    return test_accuracy, recall, precision, F1, target_num


# 日志记录函数
def log(client_idx, test_accuracy, recall, precision, F1, overall_recall, overall_precision, overall_F1):
    print("Final_Best:")
    print(f"Client {client_idx}")
    print(f"Test Accuracy: {test_accuracy.item()}")
    print(f"Class-wise Recall: {recall.squeeze()}")
    print(f"Class-wise Precision: {precision.squeeze()}")
    print(f"Class-wise F1: {F1.squeeze()}")
    print(f"Overall Recall: {overall_recall.item()}")
    print(f"Overall Precision: {overall_precision.item()}")
    print(f"Overall F1: {overall_F1.item()}")


# 加载模型并测试
save_dir = r'D:\HuYan\BT\savemodel\al=0.1'

for i in range(1, 5):
    # 创建模型
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

    # model = TumorClassifier()
    model = timm.create_model('densenet121', pretrained=False, num_classes=4)
    # model = timm.create_model('resnet50', pretrained=False, num_classes=4)
    # model = timm.create_model('mobilenetv2_100', pretrained=False, num_classes=4)
    # model = timm.create_model('convnext_base', pretrained=False, num_classes=4)
    # model = timm.create_model('edgenext_base.in21k_ft_in1k', pretrained=False, num_classes=4)
    model = model.to(device)

    # 加载模型参数
    # model_path = os.path.join(save_dir, f'client{i}_best_val.pth')
    model_path = os.path.join(save_dir, f'client{i}_best_model.pth')
    # model_path = os.path.join(save_dir, f'client{i}.pth')
    model.load_state_dict(torch.load(model_path))

    # 获取对应客户端的测试集
    test_loader = clients[i - 1]['test']

    # 进行测试
    test_accuracy, recall, precision, F1, target_num = test(model, test_loader)

    # 过滤掉样本数为 0 的类别
    valid_mask = target_num > 0
    valid_recall = recall[valid_mask]
    valid_precision = precision[valid_mask]
    valid_F1 = F1[valid_mask]
    valid_target_num = target_num[valid_mask]

    # 计算客户端整体指标
    overall_recall_numerator = (valid_recall * valid_target_num).sum()
    overall_recall = overall_recall_numerator / valid_target_num.sum()

    overall_precision_numerator = (valid_precision * valid_target_num).sum()
    overall_precision = overall_precision_numerator / valid_target_num.sum()

    overall_F1_numerator = (valid_F1 * valid_target_num).sum()
    overall_F1 = overall_F1_numerator / valid_target_num.sum()

    # 记录日志
    log(i, test_accuracy, recall, precision, F1, overall_recall, overall_precision, overall_F1)