import utils
import matplotlib.pyplot as plt
from task2a import cross_entropy_loss, pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer, calculate_accuracy


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # // INITIAL MODEL // ===========================
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss - Initial:", cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss - Initial:", cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Train accuracy - Initial:", calculate_accuracy(X_train, Y_train, model))
    print("Validation accuracy - Initial:", calculate_accuracy(X_val, Y_val, model))

    # // IMPROVED WEIGHTS MODEL // ========================
    use_improved_weight_init = True
    use_improved_sigmoid = False
    use_momentum = False

    model_iw = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_iw = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_iw, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_iw, val_history_iw = trainer_iw.train(num_epochs)

    print("\nImproved weights:")
    print("Final Train Cross Entropy Loss - Improved weights:", cross_entropy_loss(Y_train, model_iw.forward(X_train)))
    print("Final Validation Cross Entropy Loss - Improved weights:", cross_entropy_loss(Y_val, model_iw.forward(X_val)))
    print("Train accuracy - Improved weights:", calculate_accuracy(X_train, Y_train, model_iw))
    print("Validation accuracy - Improved weights:", calculate_accuracy(X_val, Y_val, model_iw))

    # // IMPROVED WEIGHTS & SIGMOID MODEL // =======================
    use_improved_weight_init = True
    use_improved_sigmoid = True
    use_momentum = False

    model_iws = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_iws = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_iws, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_iws, val_history_iws = trainer_iws.train(num_epochs)

    print("\nImproved weights, sigmoid:")
    print("Final Train Cross Entropy Loss - Improved weights, sigmoid:", cross_entropy_loss(Y_train, model_iws.forward(X_train)))
    print("Final Validation Cross Entropy Loss - Improved weights, sigmoid:", cross_entropy_loss(Y_val, model_iws.forward(X_val)))
    print("Train accuracy - Improved weights, sigmoid:", calculate_accuracy(X_train, Y_train, model_iws))
    print("Validation accuracy - Improved weights, sigmoid:", calculate_accuracy(X_val, Y_val, model_iws))

    # // IMPROVED WEIGHTS & SIGMOID & MOMENTUM MODEL // ===================
    use_improved_weight_init = True
    use_improved_sigmoid = True
    use_momentum = True
    learning_rate = .02

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

    print("\nImproved weights, sigmoid, momentum:")
    print("Final Train Cross Entropy Loss - Improved weights, sigmoid, momentum:", cross_entropy_loss(Y_train, model_iwsm.forward(X_train)))
    print("Final Validation Cross Entropy Loss - Improved weights, sigmoid, momentum:", cross_entropy_loss(Y_val, model_iwsm.forward(X_val)))
    print("Train accuracy - Improved weights, sigmoid, momentum:", calculate_accuracy(X_train, Y_train, model_iwsm))
    print("Validation accuracy - Improved weights, sigmoid, momentum:", calculate_accuracy(X_val, Y_val, model_iwsm))

    # // PLOT // ===================================================

    plt.figure(figsize=(20, 12))

    plt.subplot(1, 2, 1)
    plt.ylim([0, .6])
    utils.plot_loss(train_history["loss"], "Train loss - Initial", npoints_to_average=10)
    utils.plot_loss(train_history_iw["loss"], "Train loss - Improved weights", npoints_to_average=10)
    utils.plot_loss(train_history_iws["loss"], "Train loss - Improved weights, sigmoid", npoints_to_average=10)
    utils.plot_loss(train_history_iwsm["loss"], "Train loss - Improved weights, sigmoid, momentum", npoints_to_average=10)
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.ylim([0.85, .98])
    utils.plot_loss(val_history["accuracy"], "Validation accuracy - Initial")
    utils.plot_loss(val_history_iw["accuracy"], "Validation accuracy - Improved weights")
    utils.plot_loss(val_history_iws["accuracy"], "Validation accuracy - Improved weights, sigmoid")
    utils.plot_loss(val_history_iwsm["accuracy"], "Validation accuracy - Improved weights, sigmoid, momentum")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("ValidationAccuracy")
    plt.legend()

    plt.savefig("task3_improved.png")
    plt.show()
