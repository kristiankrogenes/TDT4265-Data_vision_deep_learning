import pathlib
import matplotlib.pyplot as plt
import utils
import torchvision
import torch
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10) # No need to apply softmax, as this is done in nn.CrossEntropyLoss

        for param in self.model.parameters(): # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters(): # Unfreeze the last fully-connected
            param.requires_grad = True # layer
        for param in self.model.layer4.parameters(): # Unfreeze the last 5 convolutional
            param.requires_grad = True # layers

    def forward(self, x):
        x = self.model(x)
        return x

def get_accuracy_loss_result(trainer: Trainer, name: str):

    trainer.model.eval()

    with torch.no_grad():
        train_loss, train_acc = compute_loss_and_accuracy(
            trainer.dataloader_train, trainer.model, trainer.loss_criterion
        )
        val_loss, val_acc = compute_loss_and_accuracy(
            trainer.dataloader_val, trainer.model, trainer.loss_criterion
        )
        test_loss, test_acc = compute_loss_and_accuracy(
            trainer.dataloader_test, trainer.model, trainer.loss_criterion
        )

    trainer.model.train()
    print(name)
    print(f"Training Loss: {train_loss:.2f}",
          f"Training Accuracy: {train_acc:.3f}",
          f"Validation Loss: {val_loss:.2f}",
          f"Validation Accuracy: {val_acc:.3f}",
          f"Test Loss: {test_loss:.2f}",
          f"Test Accuracy: {test_acc:.3f}",
          sep=", ")

def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)

    plt.figure(figsize=(20, 8))
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()

if __name__ == "__main__":

    utils.set_seed(0)

    epochs = 10
    batch_size = 32
    learning_rate = 5e-4
    early_stop_count = 5
    dataloaders = load_cifar10(batch_size)
    model = Model()
    
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), trainer.learning_rate)
    
    trainer.train()

    get_accuracy_loss_result(trainer, "Result for model " + str(model) + "")

    create_plots(trainer, "task4a")