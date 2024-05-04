import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


def calc_sigmoid(z):
    """
    Calculate the sigmoid of z
    """

    s = 1 / (1 + np.exp(-z))

    return s


def init_zeros(dim):
    """
    Initialize parameters w and b with zeros
    """

    w = np.zeros(shape=(dim, 1))
    b = 0

    return w, b


def fbPropagate(w, b, X, Y):
    """
    Forward and Backward propagation to compute cost and gradients
    """

    m = X.shape[1]

    # Forward propagation
    A = calc_sigmoid(np.dot(w.T, X) + b)

    # To prevent taking log(0) or log(1), add a small offset
    epsilon = 1e-10
    A = np.clip(A, epsilon, 1 - epsilon)

    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # Backward propagation
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize_params(w, b, X, Y, num_iters, learning_rate, print_cost=False):
    """
    Optimize parameters w and b using gradient descent
    """

    costs = []

    for i in range(num_iters):

        # Cost and gradient calculation
        grads, cost = fbPropagate(w, b, X, Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return w, b, costs


def predict(w, b, X):
    '''
    Predict labels using learned parameters (w, b)
    '''

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Predict probabilities
    A = calc_sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        # Convert probabilities to actual predictions
        Y_prediction[0, i] = 1 if A[0, i] >= 0.5 else 0

    return Y_prediction


def build_model(X_train, Y_train, X_test, num_iters=2000, learning_rate=0.5, print_cost=False):
    """
    Build logistic regression model
    """

    # Initialize parameters
    w, b = init_zeros(X_train.shape[0])

    # Optimize parameters
    w, b, costs = optimize_params(w, b, X_train, Y_train, num_iters, learning_rate, print_cost=False)

    # Predict train/test set examples
    Y_train_pred = predict(w, b, X_train)
    Y_test_pred = predict(w, b, X_test)

    return Y_train_pred, Y_test_pred, w, b, costs


# Load Data
train_data = pd.read_csv('allplayers.csv')
test_data = pd.read_csv('test_without_pick.csv')

Y_train = train_data['Pick']
Y_test_Index = test_data['Index'] 

features = ['Pos', 'Ht', 'Wt', 'Forty', 'Vertical', 'BenchReps', 'BroadJump', 'Cone', 'Shuttle']
train_data = train_data[features]
test_data = test_data[features]

combined = [train_data, test_data]

assert not train_data.isnull().values.any()
assert not test_data.isnull().values.any()

X_train = np.array(train_data).T
Y_train = np.array(Y_train)
Y_train = Y_train.reshape(Y_train.shape[0], 1).T
X_test = np.array(test_data).T

assert X_train.shape[1] == Y_train.shape[1]
assert X_train.shape[0] == X_test.shape[0]

Y_train_pred, Y_test_pred, w_opt, b_opt, costs = build_model(X_train, Y_train, X_test, num_iters=50000, learning_rate=0.001, print_cost=True)

# Plot learning curve
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(0.001))
plt.show()

# Save submission
submission = pd.DataFrame({
        "Index": Y_test_Index,
        "Pick": Y_test_pred.T.flatten().astype(int)
    })
submission.to_csv('submission.csv', index=False)
