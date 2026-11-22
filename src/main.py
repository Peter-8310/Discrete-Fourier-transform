import numpy as np
from discrete_fourier_transform.dft import dft, idft, idft_cont
import matplotlib.pyplot as plt 
from matplotlib.ticker import FuncFormatter, MultipleLocator

def f(t:np.ndarray):
    return t % (2*np.pi)

def graph(t, ts, x, x_idft, X_dft):
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), constrained_layout=True)
    axes[1].sharex(axes[0])
    axes[2].sharex(axes[0])
    axes[0].bar(np.arange(len(X_dft)), np.abs(X_dft), color='#ff7f0e', width=0.5)
    axes[0].set_ylabel("Normalized magnitude")
    axes[0].set_title("Magnitude Spectrum")
    axes[0].grid(True)

    axes[1].bar(np.arange(len(X_dft)), np.real(X_dft))
    axes[1].set_ylabel("Normalized real part")
    axes[1].set_title("Real Part")
    axes[1].grid(True)

    axes[2].bar(np.arange(len(X_dft)), np.imag(X_dft))
    axes[2].set_ylabel("Normalized imaginary part")
    axes[2].set_title("Imaginary Part")
    axes[2].grid(True)
    axes[2].set_xlabel("Frequency bin")
    axes[2].set_xticks(np.arange(0, len(X_dft), max(1, len(X_dft) // 10)))

    axes[3].plot(ts, x, label="Original function")
    axes[3].plot(t, np.real(x_idft), color='#ff7f0e', label="DFT reconstruction")
    axes[3].set_ylabel("Amplitude")
    axes[3].set_xlabel("Time (radians)")
    axes[3].set_title("Reconstructed Signal vs Original")
    axes[3].grid(True)
    axes[3].legend()

    plt.show()
    fig.savefig('Discretefourier transform.svg')

def main():
    fs = 100
    t = np.linspace(-2*np.pi, 2*np.pi, fs, endpoint=False)
    t2 = np.linspace(2*np.pi, 10*np.pi, fs, endpoint=False)
    
    x = f(t)

    X_dft = dft(x)

    x_cont = idft_cont(t, t2, X_dft)
    graph(t2, t, x, x_cont, X_dft/fs)

if __name__ == "__main__":
    main()