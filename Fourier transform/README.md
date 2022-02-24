from matplotlib import pyplot as plt
import numpy as np
import math

sampling_rate = 1000 # per second
sampling_interval = 1 / sampling_rate
t = np.arange(0, 1, sampling_interval)

minimum_change = 0.1
frequency_domain = np.arange(0, 50, minimum_change)
length_of_frequency_domain = len(frequency_domain)

pi = np.pi

# sinusoidal waves
freq = 2
fx = 3*np.sin(2*pi*freq*t)
freq = 5.5
fx += np.sin(2*pi*freq*t)

plt.plot(t, fx)
plt.ylabel('Amplitude')
plt.figure()

freq = 2
fx = 3*np.sin(2*pi*freq*t / minimum_change) # divide minimum_change to express a decimal place 
freq = 5.5
fx += np.sin(2*pi*freq*t / minimum_change)
#-----
N = len(fx) # Number of sampling

'''def DFT2(fx): # Using DFS matrix
    n = np.arange(N)
    k = n.reshape((N, 1)) # k : frequency domain
    e = np.exp(-2j * pi * k * n / N)
    fx = fx.reshape((N, 1))
    X = np.dot(e, fx) # DFS matrix. Nested loop can be substituted with the dot product of matrix
    return X # Return the Fourier transformed function'''

def DFT(fx):
    X_real = np.zeros(length_of_frequency_domain)
    X_imaginary = np.zeros(length_of_frequency_domain)
    X = np.zeros(length_of_frequency_domain)

    for k in range(length_of_frequency_domain):
        for n in range(N):
            X_real[k] += fx[n] * math.cos(2 * pi * k * n / N) # If k is a real number, an error occurs
            X_imaginary[k] += fx[n] * math.sin(2 * pi * k * n / N)
        X[k] = math.sqrt(X_real[k]**2 + X_imaginary[k]**2) # Pythagorean theorem

    return X # Return the Fourier transformed function

X = DFT(fx)

plt.stem(frequency_domain, X, 'b', markerfmt = ' ', basefmt = 'b')
plt.xlabel('Hz')
plt.ylabel('Amplitude')
plt.show()
