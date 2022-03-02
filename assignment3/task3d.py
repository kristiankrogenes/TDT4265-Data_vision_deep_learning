import pathlib
import matplotlib.pyplot as plt
import utils
import torch
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy


class Model(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()

        self.num_classes = num_classes

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.num_output_features = 128*4*4
        
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )


    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """

        features = self.feature_extractor(x)
        features = features.view(-1, self.num_output_features)
        output = self.classifier(features)  

        batch_size = x.shape[0]
        out = output
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out

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

def create_plots(trainer1: Trainer, trainer2: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)

    plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)
    plt.ylim([0., 3.])
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer1.train_history["loss"], label="Training loss - SGD", npoints_to_average=10)
    utils.plot_loss(trainer1.validation_history["loss"], label="Validation loss - SGD")
    utils.plot_loss(trainer2.train_history["loss"], label="Training loss - Adam", npoints_to_average=10)
    utils.plot_loss(trainer2.validation_history["loss"], label="Validation loss - Adam")
    # utils.plot_loss(trainer.test_history["loss"], label="Test loss")

    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    # utils.plot_loss(trainer.train_history["accuracy"], label="Training Accuracy")
    utils.plot_loss(trainer1.validation_history["accuracy"], label="Validation Accuracy - SGD")
    utils.plot_loss(trainer2.validation_history["accuracy"], label="Validation Accuracy - Adam")
    # utils.plot_loss(trainer.test_history["accuracy"], label="Test Accuracy")
    plt.legend()

    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()

if __name__ == "__main__":

    utils.set_seed(0)

    batch_size = 32
    learning_rate = 0.001
    model = Model(image_channels=3, num_classes=10)
    epochs = 10
    early_stop_count = 4

    dataloaders = load_cifar10(batch_size)

    sgd_trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )

    adam_trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )

    adam_trainer.optimizer = torch.optim.Adam(adam_trainer.model.parameters(), adam_trainer.learning_rate)
    
    sgd_trainer.train()
    adam_trainer.train()

    get_accuracy_loss_result(sgd_trainer, "Result for model " + str(model) + "")
    get_accuracy_loss_result(adam_trainer, "Result for model " + str(model) + "")

    create_plots(sgd_trainer, adam_trainer, "task3d_sgd_adam")
   




    