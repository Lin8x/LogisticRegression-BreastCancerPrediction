import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv("data.csv")



data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)



# converts diagnosis value of M and B to 0 or 1 to make it easy for skitlearn
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})



y = data['diagnosis'].values # makes y be an num array 



x_data = data.drop(['diagnosis'], axis=1) 
# drops diagonosis from the list so u can have all the variables needed
# for prediction on its own



# x_data – x_data.min(): 
# subtracts the minimum value from each value in the dataset 
# shifting the data so that the smallest value becomes 0.
# 
# x_data.max() – x_data.min(): 
# calculates the range of the data (difference between the 
# maximum and minimum values).
# 

x = (x_data - x_data.min()) / (x_data.max() - x_data.min())




from sklearn.model_selection import train_test_split

# splits the data between training and testing data
# it does a 15% split for the test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, random_state = 42) # cs



# This transposes the matrix so that each 
# column is one training example, and 
# each row is a feature, which aligns 
# with how matrix multiplication in 
# logistic regression is designed.

# to ensure that the data has the correct
# shape for matrix operations during the
# logistic regression.

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T
print("x train: ", x_train.shape)
print("x test: ", x_test.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)




# This function initializes the weights (w) 
# and bias (b) for a machine learning model 
# — typically a linear model like logistic 
# regression or a simple neural network layer.

# dimension - This is the number of 
# features (or input units).

# w = np.random.randn(dimension, 1) * 0.01
# This creates a weight vector w of shape (dimension, 1) filled with small random values.
# np.random.randn(dimension, 1) generates normally-distributed random numbers (mean = 0, std = 1).
# Multiplying by 0.01 makes the weights small, which helps in training (especially for gradient descent convergence).

# Why small values?
# - Prevents large initial activations.
# - Helps avoid exploding gradients in deeper networks.
# - Example: If your input data has 3 features, 
# dimension = 3.

def initialize_weights_and_bias(dimension):
    w = np.random.randn(dimension, 1) * 0.01  
    b = 0.0
    return w, b





# this is to get your value in between 0 and 1
# via the sigmoid equation
# 
# The sigmoid() function turns z (which can 
# be any number from -∞ to ∞) into a probability 
# between 0 and 1.

def sigmoid(z):
    return 1 / (1 + np.exp(-z))





# 
# Forward-Backward Propagation
#
# np.dot(w.T, x_train): Computes the matrix 
# multiplication of the weights and the input data.
#
# cost = (-1/m) * np.sum(y_train * 
# np.log(y_head) + (1 – y_train) * 
# np.log(1 – y_head)): 
# Measures the difference between the 
# predicted probability (y_head) and true 
# label (y_train).
# 
# derivative_weight = (1/m) * np.dot(x_train, 
# (y_head – y_train).T): 
# This calculates the gradient of the cost 
# with respect to the weights w. It tells us 
# how much we need to change the weights to 
# reduce the cost.
# 
# derivative_bias = (1/m) * np.sum(y_head – y_train): 
# 
# This computes the gradient of the cost with respect 
# to the bias b. It is simply the average of the 
# difference between predicted probabilities (y_head) 
# and actual labels (y_train).
# 

def forward_backward_propagation(w, b, x_train, y_train):
    
    # x_train currently looks like
    # (30, 483). This is the
    # number of rows and columns in dimension.
    
    # this equation below is to get the second value
    # (columns) of the x_train shape
    m = x_train.shape[1] 
    
    # this is a base equation to determine
    # what you want to guess while
    # taking weights and biases into
    # account
    # uses the dot function to
    # get a vector of scalar (unit) number
    # ---- According to GPT:
    #   "Performs the dot product between weights 
    #   and each training example to get a vector 
    #   of 'z' values — one for each example"
    # 
    # IE multiple transpose of weights
    # by the trained data
    #
    # this leads to an array of scalar values
    z = np.dot(w.T, x_train) + b
    
    # gets the z value of the equation results
    y_head = sigmoid(z)
    
    # binary cross-entropy loss (also called log loss) is to do the following...
    #
    # The cost line is how we measure how wrong 
    # our model is across all the training examples.
    #
    # We're training the model to predict 0 or 1 
    # (like/dislike, cat/not-cat, etc.), but it 
    # doesn’t say “1” or “0” outright — it gives 
    # a probability like “I’m 92% sure this is a yes.”
    # 
    # So we have to ask:
    #   “Okay, cool — but how bad was that guess 
    #   compared to the real answer?”
    # 
    # That’s what the cost function is for.
    #
    # y_train * log(y_head) is to determine how
    # big or small your number is in how
    # close you were and provide a result
    # that is massively or shortly negative
    # based on how "punished" you should be
    # for your answer in a positive (1) result
    #
    # if your correct answer is negative (0), instead
    # of doing the above, you will do
    # (1 - y_train) * np.log(1 - y_head)
    # this provides a negative result based on
    # how "punished" you should be for your answer
    # in a empty (0) result
    cost = (-1/m) * np.sum(y_train * np.log(y_head) + (1 - y_train) * np.log(1 - y_head))
    
    # these are the lines to actually now train
    # the model by changing its weight
    # and bias via your result
    #
    # y_head - y_train is the
    # predicted result - actual_result
    # which is used in both to see
    # how close you got in an array-form
    #
    # for derivative_weight, it is to determine
    # how correct/wrong you were based on
    # how much each feature/attribute/x value
    # was important to the result
    #
    # for derivative_bias, it is getting the average 
    # error across all examples
    derivative_weight = (1/m) * np.dot(x_train, (y_head - y_train).T)
    derivative_bias = (1/m) * np.sum(y_head - y_train)
    
    # gradients are a just the dictionary of these
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    return cost, gradients






# this update function is like an for loop
# to determine how to update the training model
# as you make predictions, get results, and improve
#
def update(w, b, x_train, y_train, learning_rate, num_iterations):
    costs = []
    gradients = {}
    # this is to do the process for however many times you want
    # you can think of each loop as one step of learning
    # the more learning, the better
    for i in range(num_iterations):
        
        # this determines how close you were to 
        # the accurate answer with a prediction
        # created via your bias and weights
        cost, grad = forward_backward_propagation(w, b, x_train, y_train)
        
        # you are applying the gradients based on
        # how fast you want the tool to learn.
        # you do this because if you just used 
        # the raw gradients (the full size of 
        # the error), you'd overshoot the solution 
        # or even cause your model to bounce around 
        # and never learn
        w -= learning_rate * grad["derivative_weight"]
        b -= learning_rate * grad["derivative_bias"]

        # this is just to save your values to determine
        # if your model is even learning in the process.
        # the cost should go down to showcase the learning
        # as your algorithim should make "less mistakes"
        # as time goes on. and therefore it's "punishment"
        # gets smaller over time.
        if i % 100 == 0:
            costs.append(cost)
            print(f"Cost after iteration {i}: {cost}")

    # parameters: your trained model — final weights & bias
    # gradients: last update info (optional, not usually needed)
    # costs: the history of cost, so you can plot learning progress
    parameters = {"weight": w, "bias": b}
    
    return parameters, gradients, costs





# predict function is the final step
# and its entire point is to ensure
# that you can now predict values
# for the test values you saved
#
def predict(w, b, x_test):
    # gets x_test to be the
    # column / example size
    m = x_test.shape[1]
    
    # create's an empty array
    # for you to make predictions
    # and update
    y_prediction = np.zeros((1, m))
    
    # you are repeating the same steps
    # during training to determine the probability
    # of your test data with the the weights
    # and bias to get an array of values
    # that is how certain your algorithim
    # thinks it is
    z = sigmoid(np.dot(w.T, x_test) + b)

    # checks within the array of z values
    # (your predictions). And if ur prediction
    # is over 50%, ur going to choose 1
    # otherwise you pick 0
    for i in range(z.shape[1]):
        y_prediction[0, i] = 1 if z[0, i] > 0.5 else 0

    return y_prediction





# this is the final algorithim using
# all our parts of the algorithim
# into 1 function called logistic regression
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate=0.01, num_iterations=1000):
    
    # dimension is how many features/attrjbutes 
    # will be applying to the equation. (ex: age, race)
    dimension = x_train.shape[0]
    # this is starting the weights with a set
    # of small values, and a bias at 0 
    w, b = initialize_weights_and_bias(dimension)
    
    # this trains the model
    parameters, gradients, costs = update(w, b, x_train, y_train, learning_rate, num_iterations)
    
    # they make predictions based on the testing and training data
    # 
    # the test on training data is to determine if it remembered 
    # the original data
    #
    # and the test on test data is to see if it
    # is actually able to make good predictions
    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)
    y_prediction_train = predict(parameters["weight"], parameters["bias"], x_train)
    
    # this prints the accuracy
    print(f"Train accuracy: {100 - np.mean(np.abs(y_prediction_train - y_train)) * 100}%")
    print(f"Test accuracy: {100 - np.mean(np.abs(y_prediction_test - y_test)) * 100}%")

logistic_regression(x_train, y_train, x_test, y_test, learning_rate=0.01, num_iterations=30000)


