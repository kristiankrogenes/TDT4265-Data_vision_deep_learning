import utils
import matplotlib.pyplot as plt
from task2a import cross_entropy_loss, pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer, calculate_accuracy

if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .02
    batch_size = 32
    neurons_per_layer = [128, 10]
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
    model_128 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_128 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_128, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_128, val_history_128 = trainer_128.train(num_epochs)

    print("\n128 hidden neurons:")
    print("Final Train Cross Entropy Loss - 128 Neurons:", cross_entropy_loss(Y_train, model_128.forward(X_train)))
    print("Final Validation Cross Entropy Loss - 128 Neurons:", cross_entropy_loss(Y_val, model_128.forward(X_val)))
    print("Train accuracy - 128 Neurons:", calculate_accuracy(X_train, Y_train, model_128))
    print("Validation accuracy - 128 Neurons:", calculate_accuracy(X_val, Y_val, model_128))

    # // PLOT // ===================================================

    plt.figure(figsize=(20, 12))

    plt.subplot(1, 2, 1)
    plt.ylim([0, .6])
    utils.plot_loss(train_history_128["loss"], "Train loss - 128 hidden neurons", npoints_to_average=10)
    utils.plot_loss(val_history_128["loss"], "Validation loss - 128 hidden neurons")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training / Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.ylim([0.85, 1.0])
    utils.plot_loss(train_history_128["accuracy"], "Train accuracy - 128 hidden neurons")
    utils.plot_loss(val_history_128["accuracy"], "Validation accuracy - 128 hidden neurons")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training / ValidationAccuracy")
    plt.legend()

    plt.savefig("task4b_128_hidden_neurons.png")
    plt.show()
