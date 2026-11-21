import numpy as np

def dft(x:np.ndarray):
    X = np.zeros(x.shape, dtype=complex)
    N = len(x)
    for n in range(0, len(x)):
        for k, v in enumerate(x):
            X[n] += v*np.exp(-1j*2*np.pi*n*k/N)
    return X

def idft(X:np.ndarray):
    x = np.zeros(X.shape, dtype=complex)
    N = len(X)
    for n in range(0, len(X)):
        for k, v in enumerate(X):
            x[n] += v*np.exp(1j*2*np.pi*n*k/N)
    return x/N