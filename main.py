import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split

from models import *
from utils import *

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import os
import re
import argparse


def get_device():
    if torch.cuda.is_available():
        print("CUDA available")
        print(f"Number of devices: {torch.cuda.device_count()}")
        for dev in range(torch.cuda.device_count()):
            print(f"Device {dev}:")
            print(f"\tName: {torch.cuda.get_device_name(dev)}")
    else:
        print("CUDA not available")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    return device


def dataset_preprocessing(validation_split, batch_size, monitor):
    val = (validation_split > 0.0)

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    full_trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )

    if val:
        train_size = int((1 - validation_split) * len(full_trainset))
        val_size = len(full_trainset) - train_size

        trainset, valset = random_split(full_trainset, [train_size, val_size])
    else:
        trainset = full_trainset


    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )
    
    
    monitor.log(f"Training full set size: {len(full_trainset)}")
    monitor.log(f"Testing set size: {len(testset)}\n")

    monitor.log(f"Training full transforms: {train_transform}")
    monitor.log(f"Testing transforms: {test_transform}\n")

    monitor.log(f"Validation split: {validation_split}")
    monitor.log(f"Batch size: {batch_size}\n")


    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2
    )

    if val:
        valloader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=False, num_workers=2
        )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    monitor.log(f"Final training set size: {len(trainloader)*batch_size} in {len(trainloader)} batches")
    monitor.log(f"Final validation set size: {len(valloader)*batch_size if val else 0} in {len(valloader) if val else 0} batches")
    monitor.log(f"Final testing set size: {len(testloader)*batch_size} in {len(testloader)} batches")
    

    return trainloader, valloader if val else None, testloader, classes


def inspect_dataset(trainloader, valloader, testloader, classes):
    def imshow(img, name):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.figure()
        plt.title(f"{name.capitalize()} sample batch")
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis("off")


    for name, loader in zip(
        ("train", "val", "test"), (trainloader, valloader, testloader)
    ):
        it = iter(loader)
        images, labels = next(it)
        imshow(torchvision.utils.make_grid(images), name)
        # print()
        # print('Labels: ' + ', '.join(f'{classes[labels[j]].strip()}' for j in range(len(labels))))

    plt.show()


def get_model(model_name, device):
    if model_name == "CustomNet":
        model = CustomNet(num_classes=10)
    elif model_name == "VGG":
        model = VGG("VGG19", num_classes=10)
    elif model_name == "ResNet18":
        model = ResNet18(num_classes=10)
    elif model_name == "PreActResNet18":
        model = PreActResNet18(num_classes=10)
    elif model_name == "LeNet":
        model = LeNet(num_classes=10)
    elif model_name == "GoogLeNet":
        model = GoogLeNet(num_classes=10)
    elif model_name == "DenseNet121":
        model = DenseNet121(num_classes=10)
    elif model_name == "ResNeXt29_2x64d":
        model = ResNeXt29_2x64d(num_classes=10)
    elif model_name == "MobileNet":
        model = MobileNet(num_classes=10)
    elif model_name == "MobileNetV2":
        model = MobileNetV2(num_classes=10)
    elif model_name == "DPN92":
        model = DPN92(num_classes=10)
    elif model_name == "ShuffleNetG2":
        model = ShuffleNetG2(num_classes=10)
    elif model_name == "SENet18":
        model = SENet18(num_classes=10)
    elif model_name == "ShuffleNetV2":
        model = ShuffleNetV2(num_classes=10)
    elif model_name == "EfficientNetB0":
        model = EfficientNetB0(num_classes=10)
    elif model_name == "RegNetX_200MF":
        model = RegNetX_200MF(num_classes=10)
    elif model_name == "SimpleDLA":
        model = SimpleDLA(num_classes=10)
    elif model_name == "DLA":
        model = DLA(num_classes=10)
    else:
        raise Exception(f"Model {model_name} doesn't exist")

    model = model.to(device)

    return model


def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)


def load_model(model, file_name):
    model.load_state_dict(torch.load(file_name))
    return model

def get_loss_function():
    return nn.CrossEntropyLoss()

def get_optimizer(model, args):
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise Exception(f"Optimizer {args.optimizer} doesn't exist")
    
    return optimizer


def get_scheduler(optimizer, args):
    if args.scheduler == "ConstantLR":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1.0
        )
    elif args.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
    elif args.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    else:
        raise Exception(f"Scheduler {args.scheduler} doesn't exist")

    return scheduler


def plot_training_metrics(
    train_losses, val_losses, train_accuracies, val_accuracies, learning_rates, model_number, base_dir
):
    fig = plt.figure()
    plt.title("Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.savefig(f"{base_dir}/plots/loss_{model_number}.pdf")
    plt.close(fig)

    fig = plt.figure()
    plt.title("Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.legend()
    plt.savefig(f"{base_dir}/plots/accuracy_{model_number}.pdf")
    plt.close(fig)

    fig = plt.figure()
    plt.title("Learning rate")
    plt.ylabel("learning rate")
    plt.xlabel("Epoch")
    plt.plot(learning_rates, label="Learning Rate")
    plt.legend()
    plt.savefig(f"{base_dir}/plots/learning_rate_{model_number}.pdf")
    plt.close(fig)



def train(model, model_number, trainloader, valloader, loss_function, optimizer, scheduler, epochs, init_epoch, patience, min_loss_improvement, device, monitor, base_dir):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    learning_rates = []

    best_val_loss = None

    for e in range(init_epoch-1, epochs):
        monitor.start(desc=f"Epoch {e + 1}/{epochs}", max_progress=len(trainloader))

        learning_rate = scheduler.get_last_lr()[0]
        learning_rates.append(learning_rate)

        train_loss = 0.0
        cumulative_loss = 0.0
        count_loss = 0

        train_accuracy = 0.0
        correct_predictions = 0
        count_predictions = 0

        model.train()
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logits = model(inputs)

            loss = loss_function(logits, labels)

            cumulative_loss += loss.item()
            count_loss += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss = cumulative_loss / count_loss

            predicted_labels = torch.argmax(logits, dim=1)
            count_predictions += labels.size(0)
            correct_predictions += (predicted_labels == labels).sum().item()
            train_accuracy = correct_predictions / count_predictions

            monitor.update(
                i + 1,
                learning_rate=f"{learning_rate:.5f}",
                train_loss=f"{train_loss:.4f}",
                train_accuracy=f"{train_accuracy:.4f}",
            )

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        monitor.stop()

        if valloader is not None:
            monitor.start(desc=f"Validation", max_progress=len(valloader))

            val_loss = 0.0
            cumulative_loss = 0.0
            count_loss = 0

            val_accuracy = 0.0
            correct_predictions = 0
            count_predictions = 0

            model.eval()
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(valloader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    logits = model(inputs)
                    loss = loss_function(logits, labels)
                    cumulative_loss += loss.item()
                    count_loss += 1
                    val_loss = cumulative_loss / count_loss
                    predicted_labels = torch.argmax(logits, dim=1)
                    count_predictions += labels.size(0)
                    correct_predictions += (predicted_labels == labels).sum().item()
                    val_accuracy = correct_predictions / count_predictions
                    monitor.update(
                        i + 1,
                        val_loss=f"{val_loss:.4f}",
                        val_accuracy=f"{val_accuracy:.4f}",
                    )

            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            monitor.stop()

            if best_val_loss is None or val_loss < best_val_loss - min_loss_improvement:
                save_model(model, f"{base_dir}/weights/best_{model_number}.pt")
                monitor.log(f"Model saved as best_{model_number}.pt\n")
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                monitor.log(f"Early stopping after {e + 1} epochs\n")
                break

        scheduler.step()

        save_model(model, f"{base_dir}/weights/last_{model_number}.pt")
        
        plot_training_metrics(
            train_losses,
            val_losses,
            train_accuracies,
            val_accuracies,
            learning_rates,
            model_number,
            base_dir,
        )

    monitor.print_stats()


def test(model, testloader, loss_function, device, monitor):
    monitor.start(desc=f"Testing", max_progress=len(testloader))

    test_loss = 0.0
    cumulative_loss = 0.0
    count_loss = 0

    test_accuracy = 0.0
    correct_predictions = 0
    count_predictions = 0

    inference_times = []

    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)

            start_time = time.perf_counter()
            logits = model(inputs)
            end_time = time.perf_counter()

            batch_inference_time = (end_time - start_time) / inputs.size(0)
            inference_times.append(batch_inference_time)

            loss = loss_function(logits, labels)
            cumulative_loss += loss.item()
            count_loss += 1
            test_loss = cumulative_loss / count_loss
            predicted_labels = torch.argmax(logits, dim=1)
            count_predictions += labels.size(0)
            correct_predictions += (predicted_labels == labels).sum().item()
            test_accuracy = correct_predictions / count_predictions
            monitor.update(
                i + 1,
                test_loss=f"{test_loss:.4f}",
                test_accuracy=f"{test_accuracy:.4f}",
            )

    monitor.stop()

    mean_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)

    monitor.log(f"Accuracy on test images: {100 * test_accuracy:.3f} %")
    monitor.log(f"Mean inference time: {mean_inference_time * 1000:.3f} ms")
    monitor.log(
        f"Standard deviation of inference time: {std_inference_time * 1000:.3f} ms"
    )


def main(args):
    torch.manual_seed(args.seed)
    device = get_device()

    model = get_model(model_name=args.model_name, device=device)

    os.makedirs("res", exist_ok=True)
    dir_name = f"{model.__class__.__name__}_{args.version}"
    if args.train and not args.resume:
        for file in os.listdir(f"res"):
            if file == dir_name:
                raise Exception(f"Directory {dir_name} already exists")

    base_dir = f"res/{dir_name}"
    sub_dirs = [base_dir, f"{base_dir}/weights", f"{base_dir}/plots"]
    for sub_dir in sub_dirs:
        os.makedirs(sub_dir, exist_ok=True)



    dataset_monitor = Monitor(
        file_name=f"{base_dir}/dataset_log.txt", resume=args.resume
    )

    trainloader, valloader, testloader, classes = dataset_preprocessing(
        validation_split=args.validation_split, batch_size=args.batch_size, monitor=dataset_monitor
    )

    # inspect_dataset(
    #     trainloader=trainloader,
    #     valloader=valloader,
    #     testloader=testloader,
    #     classes=classes,
    # )

    
    if args.train:
        train_monitor = Monitor(
            file_name=f"{base_dir}/training_log.txt", resume=args.resume
        )

        model_number = 0

        model_found = False
        pattern = r'last_(\d+)\.pt'
        for file in os.listdir(f"{base_dir}/weights"):
            match = re.match(pattern, file)
            if match:
                model_found = True
                n = int(match.group(1))
                if n > model_number:
                    model_number = n

        if model_found:
            model_number += 1

        if args.resume:
            model = load_model(model, f"{base_dir}/weights/last_{model_number-1}.pt")

        loss_function = get_loss_function()
        optimizer = get_optimizer(model, args)
        scheduler = get_scheduler(optimizer, args)
        
        if args.resume:
            for _ in range(args.resume_epoch-1):
                scheduler.step()

        train_monitor.log(f"Model:\n{model}\n")
        train_monitor.log(f"Loss function:\n{loss_function}\n")
        train_monitor.log(f"Optimizer:\n{optimizer}\n")
        train_monitor.log(f"Scheduler:\n{scheduler.__class__.__name__}")
        for attr in dir(scheduler):
            if not attr.startswith("_") and not callable(getattr(scheduler, attr)):
                train_monitor.log(f"{attr}: {getattr(scheduler, attr)}")
        train_monitor.log("\n")

        train(
            model=model,
            model_number=model_number,
            trainloader=trainloader,
            valloader=valloader,
            loss_function=loss_function,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=args.epochs,
            init_epoch=args.resume_epoch,
            patience=args.patience,
            min_loss_improvement=args.min_loss_improvement,
            device=device,
            monitor=train_monitor,
            base_dir=base_dir
        )

    if args.test:
        test_monitor = Monitor(
            file_name=f"{base_dir}/testing_log.txt", resume=True
        )
        model = load_model(model, f"{base_dir}/weights/{args.test_model_file}")
        loss_function = get_loss_function()

        test_monitor.log(f"Testing model: {args.test_model_file}")
        
        test(
            model=model,
            testloader=testloader,
            loss_function=loss_function,
            device=device,
            monitor=test_monitor,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR10 Classification")

    allowed_models = [
        "CustomNet",
        "VGG",
        "ResNet18",
        "PreActResNet18",
        "LeNet",
        "GoogLeNet",
        "DenseNet121",
        "ResNeXt29_2x64d",
        "MobileNet",
        "MobileNetV2",
        "DPN92",
        "ShuffleNetG2",
        "SENet18",
        "ShuffleNetV2",
        "EfficientNetB0",
        "RegNetX_200MF",
        "SimpleDLA",
        "DLA",
    ]

    allowed_optimizers = ["Adam", "SGD"]

    allowed_schedulers = ["ConstantLR", "StepLR", "CosineAnnealingLR"]

    parser.add_argument("--train", action="store_true", help="Enable training mode")
    parser.add_argument("--test", action="store_true", help="Enable testing mode")
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from specified model"
    )
    parser.add_argument(
        "--resume_epoch",
        type=int,
        default=1,
        help=f"Specify the epoch to resume.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=allowed_models,
        required=True,
        help=f"Specify the model name.",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=0,
        help=f"Specify the version.",
    )
    parser.add_argument(
        "--test_model_file",
        type=str,
        default="best.pt",
        help=f"Specify the model file name for testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help=f"Specify the random seed.",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.2,
        help=f"Specify the validation split.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help=f"Specify the batch size.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=allowed_optimizers,
        default="SGD",
        help=f"Specify the optimizer.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=allowed_schedulers,
        default="CosineAnnealingLR",
        help=f"Specify the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help=f"Specify the learning rate.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help=f"Specify the momentum.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0005,
        help=f"Specify the weight decay.",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=10,
        help=f"Specify the step size.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help=f"Specify gamma.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help=f"Specify the number of training epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help=f"Specify the number of epochs necessary for early stopping if there isn't improvement.",
    )
    parser.add_argument(
        "--min_loss_improvement",
        type=float,
        default=0.001,
        help=f"Specify the minimum loss difference to detect an improvement.",
    )

    args = parser.parse_args()
    main(args)
