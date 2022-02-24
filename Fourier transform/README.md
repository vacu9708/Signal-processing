# Fourier transform
>A Fourier transform is a method to find the frequencies of a function.
>The time domain is transformed to frequency domain.

## Continuous time fourier transform
![image](https://user-images.githubusercontent.com/67142421/155603554-7edd2873-0942-4465-a931-b6f07a5494da.png)

For a computer, a discrete time Fourier transform is performed to analyze a function in the frequency domain.

## Discrete time Fourier Transform
![image](https://user-images.githubusercontent.com/67142421/155603855-35d16458-2fec-49c5-80d4-88d247fb215f.png)<br>
### By Euler's formula :
![image](https://user-images.githubusercontent.com/67142421/155604064-dac589d7-b367-4648-9202-df41ea56f8be.png)

~~~Python
from matplotlib import pyplot as plt
import numpy as np
import math

# Frequency domain
precision = 0.1
frequency_domain = np.arange(0, 50, precision)
length_of_frequency_domain = len(frequency_domain)
#-----
pi = np.pi

# Sinusoidal waves
sampling_rate = 100 # per second
sampling_interval = 1 / sampling_rate
t = np.arange(0, 1 / precision, sampling_interval) # The more time the signal is measured, the more prcise the transform is

freq = 2
fx = 3*np.sin(2*pi*freq*t)
freq = 5.5
fx += np.sin(2*pi*freq*t)

plt.plot(t, fx)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.figure()
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
            X_real[k] += fx[n] * math.cos(2 * pi * k * n / N) # If k is a real number, it doesn't work
            X_imaginary[k] += fx[n] * math.sin(2 * pi * k * n / N)
        X[k] = math.sqrt(X_real[k]**2 + X_imaginary[k]**2) # Pythagorean theorem

    return X # Return the Fourier transformed function

X = DFT(fx) / N # Divide N to prevent the amplitude from being too big.

plt.stem(frequency_domain, X, 'b', markerfmt = ' ', basefmt = 'b')
#plt.plot(frequency_domain, X)
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.show()
~~~
## Output
![image](https://user-images.githubusercontent.com/67142421/155615284-60108f35-f59c-4db5-866c-498c3b3e11e8.png)

# Fast Fourier Transform
> The Discrete Fourier Transform takes **O(n^2)** time because it has a nested loop, that is, it is slow.
> The Fast Fourier transform was made to solve this problem and is a essential core of the modern technology.

![image](https://user-images.githubusercontent.com/67142421/155605699-0773c7d0-99fa-4773-ac15-3ddf48958146.png)

~~~Python
~~~

## Output
