import matplotlib.pyplot as plt


def training_curve_plot(title, train_costs, test_costs, train_accuracy, test_accuracy, batch_size, learning_rate, epochs, time_taken=None):
    """
    Plot the training and test curves of a neural network over epochs for cost and accuracy

    Args:
        title (str): Title of the plot
        train_costs (list): Training costs over epochs
        test_costs (list): Test costs over epochs
        train_accuracy (list): Training accuracy over epochs
        test_accuracy (list): Test accuracy over epochs
        batch_size (int): Size of mini-batches (optional)
        learning_rate (float): Learning rate for gradient descent (optional)
        epochs (int): Number of epochs (optional)
        time_taken (float): Time taken for training in seconds (optional)
    """
    lg=18
    md=13
    sm=9

    minutes = time_taken // 60
    seconds = time_taken % 60
    training_time = f"{minutes:.0f}min {seconds:.0f}sec"
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, y=1.15, fontsize=lg)
    sub = f'| Batch size: {batch_size} | Learning rate: {learning_rate} | Number of Epochs: {epochs} | Training time: {training_time} |'
    fig.text(0.5, 0.99, sub, ha='center', fontsize=md)
    x = range(1, len(train_costs)+1)

    axs[0].plot(x, train_costs, label=f'Final train cost: {train_costs[-1]:.4f}')
    axs[0].plot(x, test_costs, label=f'Final test cost: {test_costs[-1]:.4f}')
    axs[0].set_title('Cost', fontsize=md)
    axs[0].set_xlabel('Epochs', fontsize=md)
    axs[0].set_ylabel('Cost', fontsize=md)
    axs[0].legend(fontsize=sm)
    axs[0].tick_params(axis='both', labelsize=sm)

    axs[1].plot(x, train_accuracy, label=f'Final train accuracy: {100*train_accuracy[-1]:.2f}%')
    axs[1].plot(x, test_accuracy, label=f'Final test accuracy: {100*test_accuracy[-1]:.2f}%')
    axs[1].set_title('Accuracy', fontsize=md)
    axs[1].set_xlabel('Epochs', fontsize=md)
    axs[1].set_ylabel('Accuracy (%)', fontsize=sm)
    axs[1].legend(fontsize=sm)
    axs[1].tick_params(axis='both', labelsize=sm)


def plot_weights(weights, title):
    """
    Plot the weights of a neural network as 28x28 images

    Args:
        weights (array): Weight matrices
        title (str): Title of the plot
    """
    # Reshape each row into a 28x28 image
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    fig.suptitle(title, fontsize=16)

    for i, ax in enumerate(axes.flat):
        # Reshape to 28x28
        weight_image = weights[i].reshape(28, 28)
        # Display as grayscale image
        ax.imshow(weight_image, cmap="gray")
        ax.set_title(f"Class {i}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
