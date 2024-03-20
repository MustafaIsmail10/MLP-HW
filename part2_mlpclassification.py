import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

def loss_function(predictions, labels):
    """
    The loss function is the cross-entropy loss function
    """
    temp = torch.mul(labels, torch.log(predictions))
    loss = torch.mean(-torch.sum(temp, dim=1))
    return loss

def soft_max_function(x):
    """
    Softmax function is used to obtain posterior probability distribution over the output classes
    """
    exp_x = torch.exp(x)
    summation = torch.sum(exp_x, dim=1, keepdim=True)
    result = exp_x / summation
    return result

# this function returns the neural network output for a given dataset and set of parameters
def forward_pass(w1, b1, w2, b2, input_data):
    """
    The network consists of 3 inputs, 16 hidden units, and 3 output units
    The activation function of the hidden layer is sigmoid.
    The output layer should apply the softmax function to obtain posterior probability distribution. And the function should return this distribution
    Here you are expected to perform all the required operations for a forward pass over the network with the given dataset
    """
    hidden_layer_output = torch.sigmoid(torch.matmul(input_data, w1) + b1)
    output_layer_output = torch.matmul(hidden_layer_output, w2) + b2

    # Here you are expected to apply the softmax function to obtain the posterior probability distribution over the output classes
    posterior_probability_distribution = soft_max_function(output_layer_output)
    return posterior_probability_distribution


# we load all training, validation, and test datasets for the classification task
train_dataset, train_label = pickle.load(
    open("data/part2_classification_train.data", "rb")
)
validation_dataset, validation_label = pickle.load(
    open("data/part2_classification_validation.data", "rb")
)
test_dataset, test_label = pickle.load(
    open("data/part2_classification_test.data", "rb")
)

# when you inspect the training dataset, you are going to see that the class instances are sequential (e.g [1,1,1,1 ... 2,2,2,2,2 ... 3,3,3,3])
# we shuffle the training dataset by preserving instance-label relationship
indices = list(range(len(train_dataset)))
np.random.shuffle(indices)
train_dataset = np.array([train_dataset[i] for i in indices], dtype=np.float32)
train_label = np.array([train_label[i] for i in indices], dtype=np.float32)

# In order to be able to work with Pytorch, all datasets (and labels/ground truth) should be converted into a tensor
# since the datasets are already available as numpy arrays, we simply create tensors from them via torch.from_numpy()
train_dataset = torch.from_numpy(train_dataset)
train_label = torch.from_numpy(train_label)

validation_dataset = torch.from_numpy(validation_dataset)
validation_label = torch.from_numpy(validation_label).to(torch.float32)

test_dataset = torch.from_numpy(test_dataset)
test_label = torch.from_numpy(test_label).to(torch.float32)

# You are expected to create and initialize the parameters of the network
# Please do not forget to specify requires_grad=True for all parameters since they need to be trainable.

# w1 defines the parameters between the input layer and the hidden layer
# Here you are expected to initialize w1 via the Normal distribution (mean=0, std=1).
w1 = torch.from_numpy(
    np.random.normal(0, 1, (3, 16)).astype(np.float32)
).requires_grad_(True)

# b defines the bias parameters for the hidden layer
# Here you are expected to initialize b1 via the Normal distribution (mean=0, std=1).
b1 = torch.from_numpy(
    np.random.normal(0, 1, (1, 16)).astype(np.float32)
).requires_grad_(True)

# w2 defines the parameters between the hidden layer and the output layer
# Here you are expected to initialize w2 via the Normal distribution (mean=0, std=1).
w2 = torch.from_numpy(
    np.random.normal(0, 1, (16, 3)).astype(np.float32)
).requires_grad_(True)

# and finally, b2 defines the bias parameters for the output layer
# Here you are expected to initialize b2 via the Normal distribution (mean=0, std=1).
b2 = torch.from_numpy(np.random.normal(0, 1, (1, 3)).astype(np.float32)).requires_grad_(
    True
)


# you are expected to use the stochastic gradient descent optimizer
# w1, b1, w2 and b2 are the trainable parameters of the neural network
optimizer = torch.optim.SGD([w1, b1, w2, b2], lr=0.001)

# These arrays will store the loss values incurred at every training iteration
iteration_array = []
train_loss_array = []
validation_loss_array = []


# We are going to perform the backpropagation algorithm 'ITERATION' times over the training dataset
# After each pass, we are calculating the average/mean cross entropy loss over the validation dataset along with accuracy scores on both datasets.
ITERATION = 15000
for iteration in range(1, ITERATION + 1):
    iteration_array.append(iteration)

    # we need to zero all the stored gradient values calculated from the previous backpropagation step.
    optimizer.zero_grad()

    # Using the forward_pass function, we are performing a forward pass over the network with the training data
    train_predictions = forward_pass(w1, b1, w2, b2, train_dataset)

    # here you are expected to calculate the MEAN cross-entropy loss with respect to the network predictions and the training label
    train_mean_crossentropy_loss = loss_function(train_predictions, train_label)
    train_loss_array.append(train_mean_crossentropy_loss.item())

    # We initiate the gradient calculation procedure to get gradient values with respect to the calculated loss
    train_mean_crossentropy_loss.backward()
    # After the gradient calculation, we update the neural network parameters with the calculated gradients.
    optimizer.step()

    # after each epoch on the training data we are calculating the loss and accuracy scores on the validation dataset
    # with torch.no_grad() disables gradient operations, since during testing the validation dataset, we don't need to perform any gradient operations
    with torch.no_grad():
        # Here you are expected to calculate the accuracy score on the training dataset by using the training labels.
        number_of_traning_examples = train_predictions.shape[0]
        truth_classes_indcies = torch.argmax(train_label, dim=1)
        predicted_classes_indcies = torch.argmax(train_predictions, dim=1)
        number_of_classes_predicted_correctly = torch.sum(truth_classes_indcies == predicted_classes_indcies).item()
        train_accuracy = number_of_classes_predicted_correctly / number_of_traning_examples

        validation_predictions = forward_pass(w1, b1, w2, b2, validation_dataset)

        # Here you are expected to calculate the average/mean cross entropy loss for the validation datasets by using the validation dataset labels.
        validation_mean_crossentropy_loss = loss_function(
            validation_predictions, validation_label
        )

        validation_loss_array.append(validation_mean_crossentropy_loss.item())

        # Similarly, here, you are expected to calculate the accuracy score on the validation dataset
        number_of_validation_examples = validation_predictions.shape[0]
        truth_classes_indcies = torch.argmax(validation_label, dim=1)
        predicted_classes_indcies = torch.argmax(validation_predictions, dim=1)
        number_of_classes_predicted_correctly = torch.sum(truth_classes_indcies == predicted_classes_indcies).item()
        validation_accuracy = number_of_classes_predicted_correctly / number_of_validation_examples


    print(
        "Iteration : %d - Train Loss %.4f - Train Accuracy : %.2f - Validation Loss : %.4f Validation Accuracy : %.2f"
        % (
            iteration + 1,
            train_mean_crossentropy_loss.item(),
            train_accuracy,
            validation_mean_crossentropy_loss.item(),
            validation_accuracy,
        )
    )


# after completing the training, we calculate our network's accuracy score on the test dataset...
# Again, here we don't need to perform any gradient-related operations, so we are using torch.no_grad() function.
with torch.no_grad():
    test_predictions = forward_pass(w1, b1, w2, b2, test_dataset)
    # Here you are expected to calculate the network accuracy score on the test dataset...
    number_of_test_examples = test_predictions.shape[0]
    truth_classes_indcies = torch.argmax(test_label, dim=1)
    predicted_classes_indcies = torch.argmax(test_predictions, dim=1)
    number_of_classes_predicted_correctly = torch.sum(truth_classes_indcies == predicted_classes_indcies).item()
    test_accuracy = number_of_classes_predicted_correctly / number_of_test_examples
    print("Test accuracy : %.2f" % (test_accuracy))

# We plot the loss versus iteration graph for both datasets (training and validation)
plt.plot(iteration_array, train_loss_array, label="Train Loss")
plt.plot(iteration_array, validation_loss_array, label="Validation Loss")
plt.legend()
plt.show()
