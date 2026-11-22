import numpy as np
import matplotlib.pyplot as plt  # type: ignore[reportMissingModuleSource]

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

def f(t:np.ndarray):
    return t % 2*np.pi

def main():
    fs = 1000
    t = np.linspace(-2*np.pi, 2*np.pi, fs, endpoint=False)
    
    x = f(t)

    X_dft = dft(x)
    X_normalized = X_dft / len(X_dft)
    #X_fft = np.fft.fft(x)

    x_idft = idft(X_dft)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), constrained_layout=True)
    axes[1].sharex(axes[0])
    axes[2].sharex(axes[0])
    axes[0].bar(np.arange(len(X_normalized)), np.abs(X_normalized), color='#ff7f0e', width=0.5)
    axes[0].set_ylabel("Normalized magnitude")
    axes[0].set_title("Magnitude Spectrum")
    axes[0].grid(True)

    axes[1].bar(np.arange(len(X_normalized)), np.real(X_normalized))
    axes[1].set_ylabel("Normalized real part")
    axes[1].set_title("Real Part")
    axes[1].grid(True)

    axes[2].bar(np.arange(len(X_normalized)), np.imag(X_normalized))
    axes[2].set_ylabel("Normalized imaginary part")
    axes[2].set_title("Imaginary Part")
    axes[2].grid(True)
    axes[2].set_xlabel("Frequency bin")
    axes[2].set_xticks(np.arange(0, len(X_dft), max(1, len(X_dft) // 10)))

    axes[3].plot(t, np.abs(x_idft), color='#ff7f0e')
    axes[3].set_ylabel("Amplitude")
    axes[3].set_xlabel("Time (radians)")
    axes[3].set_title("Reconstructed Signal")
    axes[3].grid(True)

    plt.show()

if __name__ == "__main__":
    main()