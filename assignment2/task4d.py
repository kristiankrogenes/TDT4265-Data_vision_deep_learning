import utils
import matplotlib.pyplot as plt
from task2a import cross_entropy_loss, pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer, calculate_accuracy

if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .02
    batch_size = 32
    neurons_per_layer = [60, 60, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_weight_init = True
    use_improved_sigmoid = True
    use_momentum = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # // 60 UNITS IN TWO HIDDEN LAYERS // ===================
    model_4d = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_4d = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_4d, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_4d, val_history_4d = trainer_4d.train(num_epochs)

    print("\n60 hidden units in both hidden layers:")
    print("Final Train Cross Entropy Loss:", cross_entropy_loss(Y_train, model_4d.forward(X_train)))
    print("Final Validation Cross Entropy Loss", cross_entropy_loss(Y_val, model_4d.forward(X_val)))
    print("Train accuracy", calculate_accuracy(X_train, Y_train, model_4d))
    print("Validation accuracy ", calculate_accuracy(X_val, Y_val, model_4d))

    # // PLOT // ===================================================

    plt.figure(figsize=(20, 12))

    plt.subplot(1, 2, 1)
    plt.ylim([0, .4])
    utils.plot_loss(train_history_4d["loss"], "Train loss", npoints_to_average=10)
    utils.plot_loss(val_history_4d["loss"], "Validation loss")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training / Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.ylim([.89, 1.0])
    utils.plot_loss(train_history_4d["accuracy"], "Train accuracy")
    utils.plot_loss(val_history_4d["accuracy"], "Validation accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training / ValidationAccuracy")
    plt.legend()

    plt.savefig("task4d.png")
    plt.show()
