import utils
import matplotlib.pyplot as plt
from task2a import cross_entropy_loss, pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer, calculate_accuracy

if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .02
    batch_size = 32
    neurons_per_layer = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10]
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

    # // 64 UNITS IN 10 HIDDEN LAYERS // ===================
    model_4e = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_4e = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_4e, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_4e, val_history_4e = trainer_4e.train(num_epochs)

    print("\n64 hidden units in both hidden layers:")
    print("Final Train Cross Entropy Loss - Model task 4e:", cross_entropy_loss(Y_train, model_4e.forward(X_train)))
    print("Final Validation Cross Entropy Loss - Model task 4e:", cross_entropy_loss(Y_val, model_4e.forward(X_val)))
    print("Train accuracy - Model task 4e:", calculate_accuracy(X_train, Y_train, model_4e))
    print("Validation accuracy - Model task 4e:", calculate_accuracy(X_val, Y_val, model_4e))

    # // MODEL FROM TASK 3 // ====================================
    neurons_per_layer = [64, 10]

    model_iwsm = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_iwsm = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_iwsm, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_iwsm, val_history_iwsm = trainer_iwsm.train(num_epochs)

    print("\nMODEL FROM TASK 3:")
    print("Final Train Cross Entropy Loss - Model task 3:", cross_entropy_loss(Y_train, model_iwsm.forward(X_train)))
    print("Final Validation Cross Entropy Loss - Model task 3:", cross_entropy_loss(Y_val, model_iwsm.forward(X_val)))
    print("Train accuracy - Model task 3:", calculate_accuracy(X_train, Y_train, model_iwsm))
    print("Validation accuracy - Model task 3:", calculate_accuracy(X_val, Y_val, model_iwsm))

    # // 60 UNITS IN TWO HIDDEN LAYERS // ===================
    neurons_per_layer = [60, 60, 10]

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
    utils.plot_loss(train_history_4d["loss"], "Train loss - Task 4d", npoints_to_average=10)
    utils.plot_loss(val_history_4d["loss"], "Validation loss- Task 4d")
    utils.plot_loss(train_history_4e["loss"], "Train loss - Task 4e", npoints_to_average=10)
    utils.plot_loss(val_history_4e["loss"], "Validation loss - Task 4e")
    utils.plot_loss(train_history_iwsm["loss"], "Train loss - Task 3", npoints_to_average=10)
    utils.plot_loss(val_history_iwsm["loss"], "Validation loss - Task 3")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training / Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.ylim([.89, 1.0])
    utils.plot_loss(train_history_4d["accuracy"], "Train accuracy - Task 4d")
    utils.plot_loss(val_history_4d["accuracy"], "Validation accuracy -Task 4d")
    utils.plot_loss(train_history_4e["accuracy"], "Train accuracy - Task 4d")
    utils.plot_loss(val_history_4e["accuracy"], "Validation accuracy -Task 4d")
    utils.plot_loss(train_history_iwsm["accuracy"], "Train accuracy - Task 3")
    utils.plot_loss(val_history_iwsm["accuracy"], "Validation accuracy -Task 3")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training / ValidationAccuracy")
    plt.legend()

    plt.savefig("task4e.png")
    plt.show()
