import utils
import matplotlib.pyplot as plt
from task2a import cross_entropy_loss, pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer, calculate_accuracy

if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .02
    batch_size = 32
    neurons_per_layer = [32, 10]
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

    # // 32 HIDDEN LAYERS // ===================
    model_32 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_32 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_32, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_32, val_history_32 = trainer_32.train(num_epochs)

    print("\n32 hidden neurons:")
    print("Final Train Cross Entropy Loss - 32 Neurons:", cross_entropy_loss(Y_train, model_32.forward(X_train)))
    print("Final Validation Cross Entropy Loss - 32 Neurons:", cross_entropy_loss(Y_val, model_32.forward(X_val)))
    print("Train accuracy - 32 Neurons:", calculate_accuracy(X_train, Y_train, model_32))
    print("Validation accuracy - 32 Neurons:", calculate_accuracy(X_val, Y_val, model_32))

    # // PLOT // ===================================================

    plt.figure(figsize=(20, 12))

    plt.subplot(1, 2, 1)
    plt.ylim([0, .6])
    utils.plot_loss(train_history_32["loss"], "Train loss - 32 hidden neurons", npoints_to_average=10)
    utils.plot_loss(val_history_32["loss"], "Validation loss - 32 hidden neurons")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training / Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.ylim([0.85, .98])
    utils.plot_loss(train_history_32["accuracy"], "Train accuracy - 32 hidden neurons")
    utils.plot_loss(val_history_32["accuracy"], "Validation accuracy - 32 hidden neurons")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training / ValidationAccuracy")
    plt.legend()

    plt.savefig("task4a_32_hidden_neurons.png")
    plt.show()
