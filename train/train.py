import torch
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

import datetime
import sklearn.model_selection

transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ]
)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def setup_train_test_datasets(batch_size=16,dryrun=False):
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
    
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)

    return train_dataset,test_dataset

def setup_train_test_loaders(batch_size, dryrun=False):
  train_dataset, test_dataset = setup_train_test_datasets(
      batch_size=batch_size, dryrun=dryrun
  )
  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,
      drop_last=True,
      num_workers=2,
  )

  test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, num_workers=2)
  return train_loader, test_loader

def train_1epoch(model, train_loader, lossfun, optimizer, device):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for x, y in tqdm(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = lossfun(out, y)
        _, pred = torch.max(out.detach(), 1)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_acc += torch.sum(pred == y)

    avg_loss = total_loss / len(train_loader.dataset)
    avg_acc = total_acc / len(train_loader.dataset)
    return avg_acc, avg_loss

def validate_1epoch(model, val_loader, lossfun, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    with torch.no_grad():
        for x, y in tqdm(val_loader):
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            loss = lossfun(out.detach(), y)
            _, pred = torch.max(out, 1)

            total_loss += loss.item() * x.size(0)
            total_acc += torch.sum(pred == y)

    avg_loss = total_loss / len(val_loader.dataset)
    avg_acc = total_acc / len(val_loader.dataset)
    return avg_acc, avg_loss

def train(model, optimizer, train_loader, val_loader, n_epochs, device):
    lossfun = torch.nn.CrossEntropyLoss()
    for epoch in tqdm(range(n_epochs)):
        train_acc, train_loss = train_1epoch(
            model, train_loader, lossfun, optimizer, device
        )
        val_acc, val_loss = validate_1epoch(model, val_loader, lossfun, device)
        print(
            f"epoch={epoch}, train loss={train_loss}, train accuracy={train_acc}, val loss={val_loss}, val accuracy={val_acc}"
        )

def train_subsec(batch_size, dryrun=False, device="cuda:0",n_epochs=1):
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.to(device)

    train_loader, val_loader = setup_train_test_loaders(
        batch_size, dryrun
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=0.0001)
    n_iterations = len(train_loader) * n_epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, n_iterations
    )

    train(
        model, optimizer, train_loader, val_loader, n_epochs=n_epochs, device=device
    )

    return model

if __name__ == '__main__':
    model = train_subsec(batch_size=64,n_epochs=50,dryrun=False)
    now = datetime.datetime.now()
    filename = "/path_to_save_dir/"+ "Resnet34_model_" + now.strftime('%Y%m%d_%H%M%S') + ".pth"
    torch.save(model,filename)