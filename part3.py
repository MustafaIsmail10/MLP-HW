import torch
import torch.nn as nn
import numpy as np
import pickle
import json


class MLPModel(nn.Module):
    def __init__(
        self,
        number_of_hidden_layers,
        number_of_neurons_in_hidden_layer,
        activation_function,
    ):
        super(MLPModel, self).__init__()
        self.number_of_hidden_layers = number_of_hidden_layers - 1
        self.hidden_layers = []
        for i in range(self.number_of_hidden_layers):
            self.hidden_layers.append(
                nn.Linear(
                    number_of_neurons_in_hidden_layer, number_of_neurons_in_hidden_layer
                ).to(torch.device("cuda:0"))
            )

        self.input_layer = nn.Linear(784, number_of_neurons_in_hidden_layer).to(
            torch.device("cuda:0")
        )
        self.output_layer = nn.Linear(number_of_neurons_in_hidden_layer, 10).to(
            torch.device("cuda:0")
        )
        self.activation_function = activation_function

    def forward(self, x):
        hidden_layer_output = self.activation_function(self.input_layer(x))
        for i in range(self.number_of_hidden_layers):
            hidden_layer_output = self.activation_function(
                self.hidden_layers[i - 1](hidden_layer_output)
            )
        output_layer_output = self.output_layer(hidden_layer_output)
        return output_layer_output


# we load all the datasets of Part 3
x_train, y_train = pickle.load(open("data/mnist_train.data", "rb"))
x_validation, y_validation = pickle.load(open("data/mnist_validation.data", "rb"))
x_test, y_test = pickle.load(open("data/mnist_test.data", "rb"))

x_train = x_train / 255.0
x_train = x_train.astype(np.float32)

x_test = x_test / 255.0
x_test = x_test.astype(np.float32)

x_validation = x_validation / 255.0
x_validation = x_validation.astype(np.float32)

# and converting them into Pytorch tensors in order to be able to work with Pytorch
x_train = torch.from_numpy(x_train).to(torch.device("cuda:0"))
y_train = torch.from_numpy(y_train).to(torch.long).to(torch.device("cuda:0"))

x_validation = torch.from_numpy(x_validation).to(torch.device("cuda:0"))
y_validation = torch.from_numpy(y_validation).to(torch.long).to(torch.device("cuda:0"))

x_test = torch.from_numpy(x_test).to(torch.device("cuda:0"))
y_test = torch.from_numpy(y_test).to(torch.long).to(torch.device("cuda:0"))

combined_train_validaton_x = torch.cat((x_train, x_validation), 0)
combined_train_validaton_y = torch.cat((y_train, y_validation), 0)


def run_single_training(
    iterations,
    number_of_hidden_layers,
    num_neurons_in_hidden_layer,
    activation_function,
    learning_rate,
    training_dataset,
    training_labels,
    validation_dataset,
    validation_labels,
):
    """
    This function runs the training procedure for the MLP model for one time with the given hyperparameters
    """
    # Setting up the model
    nn_model = MLPModel(
        number_of_hidden_layers, num_neurons_in_hidden_layer, activation_function
    )
    nn_model.to(torch.device("cuda:0"))
    nn_model.train()

    # Setting up the loss function and the optimizer <Fixed hyperparameters>
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(nn_model.parameters(), learning_rate)

    # Model statistics for the given hyperparameter set
    best_validation_accuracy = 0
    best_validation_accuracy_training_loss = None
    best_validation_accuracy_validation_loss = None
    best_epoch_num = 0

    # Some needed variables for computation
    number_of_traning_examples = training_dataset.shape[0]
    number_of_validation_examples = validation_dataset.shape[0]

    for iteration in range(1, iterations + 1):
        optimizer.zero_grad()
        train_prediction = nn_model(training_dataset)
        train_loss = loss_function(train_prediction, training_labels)
        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            # Computing the accuracy score on the training dataset <No need for that>
            predicted_classes_indcies = torch.argmax(train_prediction, dim=1)
            number_of_classes_predicted_correctly = torch.sum(
                training_labels == predicted_classes_indcies
            ).item()
            train_accuracy = (
                number_of_classes_predicted_correctly / number_of_traning_examples
            )

            # Computing the loss and accuracy score on the validation dataset
            validation_predictions = nn_model(validation_dataset)
            validation_loss = loss_function(
                validation_predictions, validation_labels
            )  # no need to calculate the loss for the validation dataset

            predicted_classes_indcies = torch.argmax(validation_predictions, dim=1)
            number_of_classes_predicted_correctly = torch.sum(
                validation_labels == predicted_classes_indcies
            ).item()
            validation_accuracy = (
                number_of_classes_predicted_correctly / number_of_validation_examples
            )

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_epoch_num = iteration
                best_validation_accuracy_validation_loss = validation_loss
                best_validation_accuracy_training_loss = train_loss
                best_validation_accuracy_training_accuracy = train_accuracy

            if best_epoch_num + 200 < iteration:
                break

    return (
        best_validation_accuracy,
        best_validation_accuracy_validation_loss.item(),
        best_validation_accuracy_training_accuracy,
        best_validation_accuracy_training_loss.item(),
    )


def run_multiple_trainings_compute_statistics(
    training_count,
    iterations,
    number_of_hidden_layers,
    num_neurons_in_hidden_layer,
    activation_function,
    learning_rate,
    training_dataset,
    training_labels,
    validation_dataset,
    validation_labels,
):
    """
    This functions runs the training procedure for the MLP model for training_count times with the given hyperparameters and computes the mean and standard deviation of the accuracy scores
    It returns the mean and interval of the accuracy scores as a tuple (mean, interval) such as [mean - interval, mean + interval]
    """
    accuracies = []
    for i in range(training_count):
        single_run_result = run_single_training(
            iterations,
            number_of_hidden_layers,
            num_neurons_in_hidden_layer,
            activation_function,
            learning_rate,
            training_dataset,
            training_labels,
            validation_dataset,
            validation_labels,
        )
        accuracies.append(single_run_result[0])

        print(
            "Run : %d - Validation Accuracy : %f - Validation Loss %f - Training Accuracy %f - Training Loss %f"
            % (
                i + 1,
                single_run_result[0],
                single_run_result[1],
                single_run_result[2],
                single_run_result[3],
            )
        )

    accuracies = np.array(accuracies, dtype=np.float32)
    standard_deviation = np.std(accuracies)
    mean = np.mean(accuracies)
    interval = 1.96 * standard_deviation / np.sqrt(training_count)
    return mean, interval


def main():
    print("Welcome to Part 3 - Charachter Classification using MLPs")
    iterations_num = 3000
    learning_rates = [0.0001, 0.001]
    hidden_layers_lst = [2, 3]
    neurons_in_hidden_layer_lst = [64, 128]
    activation_functions = [nn.LeakyReLU(), nn.Sigmoid()]
    print("The maximum number of iterations is set to %d" % iterations_num)
    config_num = 0
    means = []
    intervals = []
    configs = {}
    for leaning_rate in learning_rates:
        for hidden_layer_num in hidden_layers_lst:
            for neurons_num_in_hidden_layer in neurons_in_hidden_layer_lst:
                for activation_function in activation_functions:
                    print(
                        "------------------------------------------------------------"
                    )
                    print("Configuration: %d" % (config_num))
                    print("Leaning rate : %f" % (leaning_rate))
                    print("Number of hidden layers : %d" % (hidden_layer_num))
                    print(
                        "Number of neurons in hidden layer : %d"
                        % (neurons_num_in_hidden_layer)
                    )
                    print("Activation function : %s" % (activation_function))
                    configs[config_num] = {
                        "leaning_rate": leaning_rate,
                        "hidden_layer_num": hidden_layer_num,
                        "neurons_num_in_hidden_layer": neurons_num_in_hidden_layer,
                        "activation_function": activation_functions.index(
                            activation_function
                        ),
                    }
                    results = run_multiple_trainings_compute_statistics(
                        10,
                        iterations_num,
                        hidden_layer_num,
                        neurons_num_in_hidden_layer,
                        activation_function,
                        leaning_rate,
                        x_train,
                        y_train,
                        x_validation,
                        y_validation,
                    )
                    means.append(results[0])
                    intervals.append(results[1])
                    configs[config_num]["mean"] = results[0]
                    configs[config_num]["interval"] = results[1]
                    print("Mean : %f" % (results[0]))
                    print(
                        "Interval : [%f, %f]"
                        % (results[0] - results[1], results[0] + results[1])
                    )
                    print(
                        "------------------------------------------------------------"
                    )
                    config_num += 1

    print("The results are saved in configs_results.json")
    print(configs)

    best_config_index = means.index(max(means))
    best_config = configs[best_config_index]
    print("The best configuration is:")
    print("Leaning rate : %f" % (best_config["leaning_rate"]))
    print("Number of hidden layers : %d" % (best_config["hidden_layer_num"]))
    print(
        "Number of neurons in hidden layer : %d"
        % (best_config["neurons_num_in_hidden_layer"])
    )
    print(
        "Activation function : %s"
        % (activation_functions[best_config["activation_function"]])
    )
    print("Mean : %f" % (best_config["mean"]))
    print(
        "Interval : [%f, %f]"
        % (
            best_config["mean"] - best_config["interval"],
            best_config["mean"] + best_config["interval"],
        )
    )
    print("------------------------------------------------------------")
    results = run_multiple_trainings_compute_statistics(
        10,
        iterations_num,
        best_config["hidden_layer_num"],
        best_config["neurons_num_in_hidden_layer"],
        activation_functions[best_config["activation_function"]],
        best_config["leaning_rate"],
        combined_train_validaton_x,
        combined_train_validaton_y,
        x_test,
        y_test,
    )

    print("The accuracy score on the test dataset is:")
    print("Mean : %f" % (results[0]))
    print("Interval : [%f, %f]" % (results[0] - results[1], results[0] + results[1]))
    print("------------------------------------------------------------")


if __name__ == "__main__":
    main()
