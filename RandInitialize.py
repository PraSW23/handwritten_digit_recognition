# RandInitialize.py
import numpy as np

def initialise(L_out, L_in):
    """
    Randomly initialize the weights of a layer with L_in
    incoming connections and L_out outgoing connections.
    """
    epsilon_init = 0.12
    return np.random.rand(L_out, L_in + 1) * 2 * epsilon_init - epsilon_init

