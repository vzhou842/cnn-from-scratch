import numpy as np

class Softmax:
  # A standard fully-connected layer with softmax activation.

  def __init__(self, input_len, nodes):
    # We divide by input_len to reduce the variance of our initial values
    self.weights = np.random.randn(input_len, nodes) / input_len
    self.biases = np.zeros(nodes)

  def forward(self, input):
    '''
    Performs a forward pass of the softmax layer using the given input.
    Returns a 1d numpy array containing the respective probability values.
    - input can be any array with any dimensions.
    '''
    self.last_input_shape = input.shape

    input = input.flatten()
    self.last_input = input

    input_len, nodes = self.weights.shape

    totals = np.dot(input, self.weights) + self.biases
    self.last_totals = totals

    exp = np.exp(totals)
    return exp / np.sum(exp, axis=0)

  def backprop(self, d_L_d_ps, learn_rate):
    # We know only 1 element of d_L_d_ps will be nonzero
    for i, g in enumerate(d_L_d_ps):
      if g == 0:
        continue

      # e^totals
      totals_exp = np.exp(self.last_totals)

      # Sum of all e^totals
      totals_exp_sum = np.sum(totals_exp)

      # Gradients of the positive probability against totals
      # dims (nodes,)
      d_pi_d_totals = -totals_exp[i] * totals_exp / (totals_exp_sum ** 2)
      d_pi_d_totals[i] = totals_exp[i] * (totals_exp_sum - totals_exp[i]) / (totals_exp_sum ** 2)

      # Gradients of totals against weights/biases/input
      d_totals_d_biases = 1
      d_totals_d_weights = self.last_input
      d_totals_d_inputs = self.weights

      # Gradients of loss against totals
      d_L_d_totals = d_L_d_ps[i] * d_pi_d_totals

      # Gradients of loss against weights/biases
      d_L_d_biases = d_L_d_totals * d_totals_d_biases
      d_L_d_weights = d_totals_d_weights[np.newaxis].T @ d_L_d_totals[np.newaxis]

      # Gradients of loss against inputs
      d_L_d_inputs = d_totals_d_inputs @ d_L_d_totals

      # Update weights / biases
      self.biases -= learn_rate * d_L_d_biases
      self.weights -= learn_rate * d_L_d_weights

      return d_L_d_inputs.reshape(self.last_input_shape)
