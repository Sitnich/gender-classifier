import os

import matplotlib.pyplot as plt
import torch

from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

root_dir = os.path.abspath("..")

import src.data.loaders as ld
import src.models.model_classes as cl


def train_n_tune(config, checkpoint_dir=None, data_dir=None):
    from ray import tune

    # загружаем обучающую и тестовую выборки

    X_train, Y_train, X_test, Y_test = ld.load_train_test()

    train_dataset = cl.MelDataset(X_train, Y_train)
    test_dataset = cl.MelDataset(X_test, Y_test)

    model = cl.Model(inp_size=X_train.shape[1])
    train_loss_all, test_loss_all = [], []

    # объявляем функцию потерь  и оптимизатор

    criterion = F.binary_cross_entropy
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)

    # загружаем информацию о предобученной модели и оптимизаторе
    # если она уже лежит в папке

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # разделяем трейн на обучающую и валидационную подвыборки

    test_abs = int(len(train_dataset) * 0.8)
    train_subset, val_subset = random_split(
        train_dataset, [test_abs, len(train_dataset) - test_abs])

    trainloader = DataLoader(train_subset, batch_size=int(config["batch_size"]), shuffle=True)
    valloader = DataLoader(val_subset, batch_size=int(config["batch_size"]), shuffle=True)

    for epoch in range(config["epochs"]):
        running_loss = 0.0
        epoch_steps = 0
        for i, (inputs, labels) in enumerate(trainloader):
            if len(inputs) < int(config["batch_size"]):
                continue
            inputs, labels = inputs, labels
            inputs = inputs.reshape((int(config["batch_size"]), 1, -1))
            labels = labels.type(torch.float)

            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # считаем бинарную кросс энтропию и выводим через каждые 2000 элементов
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # считаем лосс и аккураси на валидационной выборке
        val_loss = 0.0
        val_steps = 0
        total = len(val_subset)
        correct = 0
        for i, (inputs, labels) in enumerate(valloader):
            if len(inputs) < int(config["batch_size"]):
                continue
            with torch.no_grad():
                inputs, labels = inputs, labels
                inputs = inputs.reshape((int(config["batch_size"]), 1, -1))
                labels = labels.type(torch.float)

                outputs = model(inputs)

                predicted = torch.Tensor([1 if p >= 0.5 else 0 for p in outputs]).reshape(-1, 1)
                correct += (predicted == labels).float().sum()

                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_steps += 1

        train_loss_all.append(running_loss / epoch_steps)
        test_loss_all.append(int(val_loss) / val_steps)

        # сохраняем информацию о состоянии модели и оптимизатора

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        tune.report(loss=(int(val_loss) / val_steps), accuracy=correct / total,
                    train_list=train_loss_all, test_list=test_loss_all)
    print("Обучение закончено!")

def train_fix_param_model(config, path = root_dir + '\\reports\\train_info'+'.txt',
                          root_dir = os.path.abspath("..")):

    # загружаем обучающую и тестовую выборки

    X_train, Y_train, X_test, Y_test = ld.load_train_test(root_dir = root_dir)

    train_dataset = cl.MelDataset(X_train, Y_train)
    test_dataset = cl.MelDataset(X_test, Y_test)


    model = cl.Model(inp_size=X_train.shape[1])
    epochs, batch_size, lr = config['epochs'], config['batch_size'], config['lr']

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_loss_all, test_loss_all = [], []
    train_acc_all, test_acc_all = [], []


    for ep in range(epochs):
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
        total_batches_train, total_batches_test = 0, 0
        train_loss, test_loss = 0, 0
        train_acc, test_acc = 0, 0

        for i, (batch, label) in tqdm(enumerate(train_dataloader),
                                      total=(len(train_dataset) + batch_size) // batch_size):
            if len(batch) < batch_size:
                continue
            total_batches_train += 1

            batch = batch.reshape((batch_size, 1, 128))
            pred = model(batch)
            prob = torch.Tensor([1 if p >= 0.5 else 0 for p in pred]).reshape(-1, 1)
            label = label.type(torch.float)
            loss = F.binary_cross_entropy(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (prob == label).float().sum()

        for i, (batch, label) in enumerate(test_dataloader):
            if len(batch) < batch_size:
                continue
            total_batches_test += 1


            batch = batch.reshape((batch_size, 1, 128))
            pred = model(batch)
            prob = torch.Tensor([1 if p >= 0.5 else 0 for p in pred]).reshape(-1, 1)
            label = label.type(torch.float)
            loss = F.binary_cross_entropy(pred, label)

            test_loss += loss.item()
            test_acc += (prob == label).float().sum()

        train_loss_all.append(train_loss / total_batches_train)
        test_loss_all.append(test_loss / total_batches_train)

        train_acc_all.append(100 * train_acc / len(train_dataset))
        test_acc_all.append(100 * test_acc / len(test_dataset))

        with open(path, "a") as file:
            file.write(f"Epoch {ep + 1} | Train loss: {train_loss_all[-1]} | Test loss: {test_loss_all[-1]} | \
            Train Accuracy: {train_acc_all[-1]} | Test Accuracy: {test_acc_all[-1]} \n")
    return train_loss_all, test_loss_all, train_acc_all, test_acc_all


# функция выводящая графики для обучающей и тестовой выборок
def print_train_test(train_l, test_l, met='Loss'):
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(train_l)), train_l, label="train")
    plt.plot(range(len(test_l)), test_l, label="test")
    plt.xlabel("Epoch")
    plt.ylabel(met)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'images/'+ met + '.png')
    plt.show()