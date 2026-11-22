from torch import stack, tril
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
# import torch
from torch.nn import Parameter, Linear
from torch import tensor, tensordot, ones, matmul, zeros 
from torch.nn.functional import relu, softmax
from torch import movedim

"""
##################
### QUESTION 1 ###
##################
"""


class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'

        Hint: You can use ones(dim) to create a tensor of dimension dim.
        """
        super(PerceptronModel, self).__init__()
        self.w = Parameter(ones(1, dimensions))


    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def forward(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (dimensions)
        Returns: a node containing a single number (the score)

        The pytorch function `tensordot` may be helpful here.
        """
        return matmul(x, self.w[0])

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        return 1 if self(x).item() >= 0 else -1



class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        super().__init__()
        self.input_layer = Linear(1, 128)
        self.hidden_layer = Linear(128, 128)
        self.output_layer = Linear(128, 1)
   

    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        hidden = relu(self.input_layer(x))
        hidden = relu(self.hidden_layer(hidden))
        return self.output_layer(hidden)


class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        super().__init__()
        self.input_layer = Linear(28 * 28, 256)
        self.hidden_layer = Linear(256, 128)
        self.output_layer = Linear(128, 10)


    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        hidden = relu(self.input_layer(x))
        hidden = relu(self.hidden_layer(hidden))
        return self.output_layer(hidden)



class LanguageIDModel(Module):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        super(LanguageIDModel, self).__init__()
        self.hidden_size = 192
        self.input_layer = Linear(self.num_chars, self.hidden_size)
        self.recurrent_layer = Linear(self.hidden_size, self.hidden_size, bias=False)
        self.output_layer = Linear(self.hidden_size, len(self.languages))



    def forward(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        tensor with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a tensor that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single tensor of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a tensor of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        hidden = xs[0].new_zeros((xs[0].size(0), self.hidden_size))
        for characters in xs:
            next_hidden = relu(self.input_layer(characters) + self.recurrent_layer(hidden))
            # The dataset pads individual words with all-zero character vectors.
            hidden = next_hidden.where(characters.any(dim=-1, keepdim=True), hidden)
        return self.output_layer(hidden)



def Convolve(input: tensor, weight: tensor):
    """
    Acts as a convolution layer by applying a 2d convolution with the given inputs and weights.
    DO NOT import any pytorch methods to directly do this, the convolution must be done with only the functions
    already imported.

    There are multiple ways to complete this function. One possible solution would be to use 'tensordot'.
    If you would like to index a tensor, you can do it as such:

    tensor[y:y+height, x:x+width]

    This returns a subtensor who's first element is tensor[y,x] and has height 'height, and width 'width'

    The method 'zeros((y_dim,x_dim))' may also be useful. It initializes a pytorch tensor with dimensions (y_dim, x_dim), with every value
    set to zero.
    """
    kernel_height, kernel_width = weight.shape
    output_height = input.shape[-2] - kernel_height + 1
    output_width = input.shape[-1] - kernel_width + 1
    # Each slice gathers a single kernel offset across all output positions.
    patches = stack(
        [
            input[..., row:row + output_height, col:col + output_width]
            for row in range(kernel_height)
            for col in range(kernel_width)
        ],
        dim=-1,
    )
    return matmul(patches, weight.flatten())


class DigitConvolutionalModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    A trainable 3x3 kernel produces a feature map for a dense classifier.
    """

    def __init__(self):
        super().__init__()
        self.convolution_weights = Parameter(ones((3, 3)))
        self.input_layer = Linear(26 * 26, 256)
        self.hidden_layer = Linear(256, 128)
        self.output_layer = Linear(128, 10)


    def forward(self, x):
        """
        Convolve each image, then flatten its feature map for classification.
        """
        images = x.reshape(len(x), 28, 28)
        features = Convolve(images, self.convolution_weights).flatten(start_dim=1)
        hidden = relu(self.input_layer(features))
        hidden = relu(self.hidden_layer(hidden))
        return self.output_layer(hidden)


class Attention(Module):
    def __init__(self, layer_size, block_size):
        super().__init__()
        """
        All the layers you should use are defined here.

        In order to pass the autograder, make sure each linear layer matches up with their corresponding matrix,
        ie: use self.k_layer to generate the K matrix.
        """
        self.k_layer = Linear(layer_size, layer_size)
        self.q_layer = Linear(layer_size, layer_size)
        self.v_layer = Linear(layer_size, layer_size)

        # Masking part of attention layer
        self.register_buffer(
            "mask",
            tril(ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

        self.layer_size = layer_size

    def forward(self, input):
        """
        Applies the attention mechanism to input. All necessary layers have
        been defined in __init__()

        In order to apply the causal mask to a given matrix M, you should update
        it as such:

        M = M.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))[0]

        For the softmax activation, it should be applied to the last dimension of the input,
        Take a look at the "dim" argument of torch.nn.functional.softmax to figure out how to do this.
        """
        _, length, _ = input.size()
        queries = self.q_layer(input)
        keys = self.k_layer(input)
        values = self.v_layer(input)
        scores = matmul(queries, movedim(keys, -1, -2)) / self.layer_size**0.5
        scores = scores.masked_fill(self.mask[0, :, :length, :length] == 0, float("-inf"))
        return matmul(softmax(scores, dim=-1), values)

