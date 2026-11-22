from torch import no_grad
from torch.utils.data import DataLoader


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


"""
##################
### QUESTION 1 ###
##################
"""


def train_perceptron(model, dataset):
    """
    Train the perceptron until convergence.
    You can iterate through DataLoader in order to 
    retrieve all the batches you need to train on.

    Each sample in the dataloader is in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.
    """
    with no_grad():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        while True:
            mistakes = 0
            for batch in dataloader:
                features = batch["x"]
                label = batch["label"].item()
                if model.get_prediction(features) != label:
                    model.get_weights().add_(features, alpha=label)
                    mistakes += 1
            if mistakes == 0:
                return


def _train_epoch(model, dataloader, optimizer, loss_function, *, sequence_first=False):
    for batch in dataloader:
        inputs = batch["x"]
        if sequence_first:
            inputs = movedim(inputs, 1, 0)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(inputs), batch["label"])
        loss.backward()
        optimizer.step()


def train_regression(model, dataset):
    """
    Trains the model.

    In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
    batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

    Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.

    Inputs:
        model: Pytorch model to use
        dataset: a PyTorch dataset object containing data to be trained on
        
    """
    model.train()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    for _ in range(230):
        _train_epoch(model, dataloader, optimizer, regression_loss)


def train_digitclassifier(model, dataset):
    """
    Trains the model.
    """
    model.train()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for _ in range(12):
        _train_epoch(model, dataloader, optimizer, digitclassifier_loss)


def train_languageid(model, dataset):
    """
    Trains the model.

    Note that when you iterate through dataloader, each batch will returned as its own vector in the form
    (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
    get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
    that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
    as follows:

    movedim(input_vector, initial_dimension_position, final_dimension_position)

    For more information, look at the pytorch documentation of torch.movedim()
    """
    model.train()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for _ in range(20):
        _train_epoch(model, dataloader, optimizer, languageid_loss, sequence_first=True)



def Train_DigitConvolution(model, dataset):
    """
    Trains the model.
    """
    model.train()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for _ in range(30):
        _train_epoch(model, dataloader, optimizer, digitconvolution_Loss)
