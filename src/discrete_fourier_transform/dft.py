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

def idft_cont(t:np.ndarray,  t_eval:np.ndarray, X:np.ndarray):
    
    x=np.zeros(t_eval.shape, dtype=complex)
    N = len(X)

    for n, ts in enumerate(t_eval):
        del_t = (ts - t[0])/(t[1] - t[0])
        x[n] = X[0]

        for k in range(1, (N+1)//2):
            p = 2*np.pi*del_t*k/N
            x[n] += X[k] * np.exp(1j*p) + X[N-k] * np.exp(-1j*p)
    
        if n % 2 == 0:
            x[n] += X[N//2] * np.cos(np.pi*del_t)

    return x/N
