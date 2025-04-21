import numpy as np
from time import perf_counter
from tqdm.auto import tqdm


def initialize(K, D, D_i, D_o):
    """
    Initializes the weights and biases of a neural network

    Args:
        K (int): Number of hidden layers
        D (int): Number of neurons per layer
        D_i (int): Input layer dimension
        D_o (int): Output layer dimension
    
    Returns:
        all_weights (list): Weight matrices for each layer
        all_biases (list): Bias vectors for each layer

    """
    all_weights = [None] * (K+1)
    all_biases = [None] * (K+1)

    # Create input and output layer
    if K==0:
        all_weights[0] = np.random.normal(0, np.sqrt(2 / D_i), size=(D_o, D_i))
    else:
        all_weights[0] = np.random.normal(0, np.sqrt(2 / D_i), size=(D, D_i))
        all_weights[-1] = np.random.normal(0, np.sqrt(2 / D), size=(D_o, D))
        all_biases[0] = np.zeros((D, 1))

    all_biases[-1] = np.zeros((D_o, 1))

    # Create intermediate layers
    for layer in range(1, K):
        all_weights[layer] = np.random.normal(0, np.sqrt(2 / D), size=(D, D))
        all_biases[layer] = np.zeros((D, 1))

    return all_weights, all_biases


def ReLU(preactivation):
    """
    ReLU activation function
    
    Args:
        preactivation (array): Preactivation values

    Returns:
        activation (array): Activation values
    """
    activation = np.maximum(0, preactivation)
    return activation


def sigmoid(preactivation):
    """
    Sigmoid activation function
    
    Args:
        preactivation (array): Preactivation values

    Returns:
        activation (array): Activation values
    """
    activation = 1 / (1 + np.exp(-preactivation))
    return activation


def forward_pass(net_input, all_weights, all_biases, activation="ReLU"):
    """
    Forward pass through the neural network

    Args:
        net_input (array): Input to the network
        all_weights (list): Weight matrices for each layer
        all_biases (list): Bias vectors for each layer
        activation (str): Activation function to use ("ReLU" or "sigmoid")
    
    Returns:
        net_output (array): Output of the network
        all_f (list): Pre-activations at each layer
        all_h (list): Activations at each layer
    """
    # Retrieve number of layers
    K = len(all_weights) - 1

    # We'll store the pre-activations at each layer in a list "all_f"
    # and the activations in a second list "all_h".
    all_f = [None] * (K+1)
    all_h = [None] * (K+1)

    #For convenience, we'll set all_h[0] to be the input, and all_f[K] will be the output
    all_h[0] = net_input

    # Run through the layers, calculating all_f[0...K-1] and all_h[1...K]
    for layer in range(K):
        # Update preactivations and activations at this layer
        all_f[layer] = np.matmul(all_h[layer], all_weights[layer].T) + all_biases[layer].T
        
        # Apply the selected activation function
        if activation == "ReLU":
            all_h[layer+1] = ReLU(all_f[layer])
        elif activation == "sigmoid":
            all_h[layer+1] = sigmoid(all_f[layer])

    # Compute the output from the last hidden layer
    all_f[K] = np.matmul(all_h[K], all_weights[K].T) + all_biases[K].T

    # Retrieve the output
    net_output = all_f[K]

    return net_output, all_f, all_h


def softmax(net_output):
    """
    Softmax activation function

    Args:
        net_output (array): Net output values
    
    Returns:
        probs (array): Softmax probabilities
    """
    exp_values = np.exp(net_output - np.max(net_output, axis=1, keepdims=True))
    probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    return probs


def compute_cost(net_output, y):
    """
    Compute the cross-entropy loss

    Args:
        net_output (array): Output of the network
        y (array): True labels
    
    Returns:
        cross_entropy_loss (float): Cross-entropy loss
    """
    I = y.shape[0]

    # Compute softmax probabilities
    probs = softmax(net_output)
    
    # Cross-entropy loss
    cross_entropy_loss = -np.sum(y * np.log(probs)) / I
    
    return cross_entropy_loss


def d_cost_d_output(net_output, y):
    """
    Compute the derivative of the cross-entropy loss with respect to the output

    Args:
        net_output (array): Output of the network
        y (array): True labels

    Returns:
        (array): Derivative of the cost with respect to the output
    """
    I = y.shape[0]
    probs = softmax(net_output)
    return (probs - y) / I


def backward_pass(K, all_weights, all_biases, all_f, all_h, y):
    """
    Backward pass through the neural network

    Args:
        K (int): Number of hidden layers
        all_weights (list): Weight matrices for each layer
        all_biases (list): Bias vectors for each layer
        all_f (list): Pre-activations at each layer
        all_h (list): Activations at each layer
        y (array): True labels
    
    Returns:
        all_dl_dweights (list): Derivatives of the cost with respect to the weights
        all_dl_dbiases (list): Derivatives of the cost with respect to the biases
    """
    # We'll store the derivatives dl_dweights and dl_dbiases in lists as well
    all_dl_dweights = [None] * (K+1)
    all_dl_dbiases = [None] * (K+1)
    # And we'll store the derivatives of the cost with respect to the activation and preactivations in lists
    all_dl_df = [None] * (K+1)
    all_dl_dh = [None] * (K+1)
    # Again for convenience we'll stick with the convention that all_h[0] is the net input and all_f[k] in the net output

    # Compute derivatives of the cost with respect to the network output
    all_dl_df[K] = np.array(d_cost_d_output(all_f[K], y))
    # Now work backwards through the network
    for layer in range(K, -1, -1):
        # Calculate the derivatives of the cost with respect to the biases at layer from all_dl_df[layer]
        all_dl_dbiases[layer] = np.sum(all_dl_df[layer], axis=0, keepdims=True).T

        # Calculate the derivatives of the cost with respect to the weights at layer from all_dl_df[layer] and all_h[layer]
        all_dl_dweights[layer] = np.matmul(all_dl_df[layer].T, all_h[layer])

        # calculate the derivatives of the cost with respect to the activations from weight and derivatives of next preactivations
        all_dl_dh[layer] = np.matmul(all_weights[layer].T, all_dl_df[layer].T)

        if layer > 0:
            # Calculate the derivatives of the cost with respect to the pre-activation f
            all_dl_df[layer-1] = np.where(all_f[layer-1] > 0, 1, 0) * all_dl_dh[layer].T

    return all_dl_dweights, all_dl_dbiases


def update_parameters(all_weights, all_biases, all_dl_dweights, all_dl_dbiases, learning_rate, optimizer=None):
    """
    Updates the weights and biases using standard gradient descent, momentum or Adam
    
    Args:
        all_weights (list): Current weight matrices
        all_biases (list): Current bias vectors
        all_dl_dweights (list): Gradients for weights
        all_dl_dbiases (list): Gradients for biases
        learning_rate (float): Learning rate for the update
        optimizer (str): Optimizer to use (None, "momentum" or "Adam")
        
    Returns:
        all_weights (list): Updated weights
        all_biases (list): Updated biases
    """
    L = len(all_weights)

    if optimizer == "momentum":
        beta = 0.9
        m_dw = [np.zeros_like(w) for w in all_weights]
        m_db = [np.zeros_like(b) for b in all_biases]
    
    elif optimizer == "Adam":
        beta = 0.9
        gamma = 0.999
        epsilon = 1e-8
        m_dw = [np.zeros_like(w) for w in all_weights]
        m_db = [np.zeros_like(b) for b in all_biases]
        v_dw = [np.zeros_like(w) for w in all_weights]
        v_db = [np.zeros_like(b) for b in all_biases]

    for layer in range(L):
        if optimizer == "momentum":
            # Compute velocity updates
            m_dw[layer] = beta * m_dw[layer] + (1 - beta) * all_dl_dweights[layer]
            m_db[layer] = beta * m_db[layer] + (1 - beta) * all_dl_dbiases[layer]

            # Update weights and biases using momentum
            all_weights[layer] = all_weights[layer] - learning_rate * m_dw[layer]
            all_biases[layer] = all_biases[layer] - learning_rate * m_db[layer]

        elif optimizer == "Adam":
            # Compute mean and pointwise squared gradients with momentum
            m_dw[layer] = beta * m_dw[layer] + (1 - beta) * all_dl_dweights[layer]
            m_db[layer] = beta * m_db[layer] + (1 - beta) * all_dl_dbiases[layer]

            v_dw[layer] = gamma * v_dw[layer] + (1 - gamma) * all_dl_dweights[layer]**2
            v_db[layer] = gamma * v_db[layer] + (1 - gamma) * all_dl_dbiases[layer]**2

            # Moderate near start of the sequence
            m_dw_hat = m_dw[layer] / (1 - beta ** (layer + 1))
            m_db_hat = m_db[layer] / (1 - beta ** (layer + 1))

            v_dw_hat = v_dw[layer] / (1 - gamma ** (layer + 1))
            v_db_hat = v_db[layer] / (1 - gamma ** (layer + 1))

            # Update weights and biases using Adam
            all_weights[layer] = all_weights[layer] - learning_rate * m_dw_hat / (np.sqrt(v_dw_hat) + epsilon)
            all_biases[layer] = all_biases[layer] - learning_rate * m_db_hat / (np.sqrt(v_db_hat) + epsilon)

        else:
            # Update weights and biases using standard gradient descent
            all_weights[layer] = all_weights[layer] - learning_rate * all_dl_dweights[layer]
            all_biases[layer]  = all_biases[layer]  - learning_rate * all_dl_dbiases[layer]

    return all_weights, all_biases


def predict(net_input, y, all_weights, all_biases):
    """
    Uses the trained network to predict classes for input data
    
    Args:
        net_input (array): Input data
        y (array): True labels
        all_weights (list): Weight matrices
        all_biases (list): Bias vectors
        
    Returns:
        predictions (array): Predicted classes
        accuracy (float): Classification accuracy
        cost (float): Cross-entropy cost
    """
    # Forward pass
    net_output, _, _ = forward_pass(net_input, all_weights, all_biases)
    
    # Compute softmax probabilities
    probs = softmax(net_output)

    # Get predicted classes
    predictions = np.argmax(probs, axis=1)
    
    # Get true classes
    true_labels = np.argmax(y, axis=1)
    
    # Compute accuracy
    accuracy = np.mean(predictions == true_labels)
    
    # Compute cost
    cost = compute_cost(net_output, y)
    
    return predictions, accuracy, cost


def random_mini_batches(net_input, y, batch_size=64):
    """
    Generates random mini-batches from the input data
    
    Args:
        net_input (array): Input data
        y (array): True labels
        batch_size (int): Size of each mini-batch
        
    Returns:
        batches (list of tuples): Generated mini-batches
    """
        
    I = net_input.shape[0] # Number of examples

    # Shuffle the data
    permutation = np.random.permutation(I)
    X_shuffled = net_input[permutation, :]
    Y_shuffled = y[permutation, :]
    batches = []

    # Partition (shuffled) data into mini-batches
    num_batches = I // batch_size
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size

        X_batch = X_shuffled[start:end, :]
        Y_batch = Y_shuffled[start:end, :]

        batches.append((X_batch, Y_batch))

    # Add remaining examples that don't fit into a mini-batch (if any)
    if I % batch_size != 0:
        start = num_batches * batch_size

        X_batch = X_shuffled[start:, :]
        Y_batch = Y_shuffled[start:, :]

        batches.append((X_batch, Y_batch))

    return batches


def train_model(X_train, Y_train, model, activation, optimizer=None, epochs=100,
                learning_rate=0.01, batch_size=64, X_test=None, Y_test=None, print_every=None):
    """
    Trains the neural network model
    
    Args:
        X_train (array): Training data
        Y_train (array): Training labels (one-hot encoded)
        model (list): Architecture defined as a list [K, D]
        activation (str): Activation function to use ("ReLU" or "sigmoid")
        optimizer (str): Optimizer to use (None, "momentum" or "Adam")
        epochs (int): Number of epochs
        learning_rate (float): Learning rate
        batch_size (int): Mini-batch size
        X_test (array): Test data (optional)
        Y_test (array): Test labels (one-hot encoded) (optional)
        print_every (int): Frequency (in epochs) to print cost and accuracy
    
    Returns:
        all_weights (list): Trained weight matrices
        all_biases (list): Trained bias vectors
        train_costs (list): Training costs
        test_costs (list): Test costs
        train_accuracies (list): Training accuracies
        test_accuracies (list): Test accuracies
        time_taken (datetime.timedelta): Time taken to train the model
    """
    init_time = perf_counter()
    K, D = model
    D_i, D_o = X_train.shape[1], Y_train.shape[1]

    # Initialize weights and biases
    all_weights, all_biases = initialize(K, D, D_i, D_o)
    
    train_costs = []
    test_costs = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in tqdm(range(1, epochs+1)):
        batch_cost = []
        batch_acc = []

        mini_batches = random_mini_batches(X_train, Y_train, batch_size)

        for X_batch, Y_batch in mini_batches:
            # Forward pass
            _, all_f, all_h = forward_pass(X_batch, all_weights, all_biases, activation)

            # Backward pass
            all_dl_dweights, all_dl_dbiases = backward_pass(K, all_weights, all_biases, all_f, all_h, Y_batch)

            # Update parameters
            all_weights, all_biases = update_parameters(all_weights, all_biases, all_dl_dweights, all_dl_dbiases, learning_rate, optimizer)

            # Compute batch's cost and accuracy on training set
            _, train_acc, train_cost = predict(X_batch, Y_batch, all_weights, all_biases)
            batch_cost.append(train_cost)
            batch_acc.append(train_acc)
        
        # Compute epoch's cost and accuracy on training set
        train_costs.append(np.mean(batch_cost))
        train_accuracies.append(np.mean(batch_acc))

        # Compute epoch's cost and accuracy on test set
        if X_test is not None and Y_test is not None:
            _, test_acc, test_cost = predict(X_test, Y_test, all_weights, all_biases)
            test_costs.append(test_cost)
            test_accuracies.append(test_acc)

        if print_every:
            # Print cost and accuracy every few epochs
            if epoch % print_every == 0 or epoch == epochs - 1:
                if X_test is not None and Y_test is not None:
                    # Print cost and accuracy
                    print(f"Epoch: {epoch} | Train Loss: {train_costs[epoch-1]:.2f} | Train Acc: {train_accuracies[epoch-1]:.2f} | Test Loss: {test_cost:.2f} | Test Acc: {test_acc:.2f}")
                else:
                    print(f"Epoch: {epoch} | Train Loss: {train_costs[epoch-1]:.2f} | Train Acc: {train_accuracies[epoch-1]:.2f}")

    # Print time taken
    time_taken = perf_counter() - init_time
    print(f"Time taken: {time_taken}")
    
    return all_weights, all_biases, train_costs, test_costs, train_accuracies, test_accuracies, time_taken
