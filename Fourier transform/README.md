# Fourier transform
![image](https://user-images.githubusercontent.com/67142421/155687402-a9ae5d4a-9baa-4a83-ac6e-b504ebf805df.png)
>A Fourier transform is a method to find the frequencies of a function. The time domain is transformed to frequency domain.<br>
>The amplitude of a frequency is a complex number.

## Continuous time fourier transform (ω = 2πf)
![image](https://user-images.githubusercontent.com/67142421/155603554-7edd2873-0942-4465-a931-b6f07a5494da.png)<br>
For a computer, a discrete time Fourier transform is performed to analyze a function in the frequency domain.

## Discrete time Fourier Transform (Induced by Riemann sum integral)
![image](https://user-images.githubusercontent.com/67142421/155689010-f04e9a51-ccba-4951-81d2-6346de16f5fc.png)

![image](https://user-images.githubusercontent.com/67142421/155687366-75207445-8ab9-49fe-9505-6c11786e877f.png)<br>
* The infinitesimal dt is 1 because it is discrete time.
* n(sample) corresponds to t(time).
* k corresponds to f (k'th frequency in the frequency domain)

### By Euler's formula :
![image](https://user-images.githubusercontent.com/67142421/155604064-dac589d7-b367-4648-9202-df41ea56f8be.png)

### Characteristics
* maximum frequency = sampling frequency / 2
* The longer time the signal is measured, the better the frequency resolution is. 
  For example : to measure 1 Hz, the signal has to be recorded for 1 second and to measure 0.1 Hz, the signal has to be recorded for 10 seconds.

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
freq = 7.4
fx += np.sin(2*pi*freq*t)

plt.plot(t, fx)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.figure()
#-----
N = len(fx) # Number of sampling

'''def DFT2(fx): # Using DFS matrix
    n = np.arange(N)
    k = frequency_domain.reshape((length_of_frequency_domain, 1))
    e = np.exp(-2j * pi * k * n / N)
    fx = fx.reshape((len(fx), 1))
    X = np.dot(e, fx) # Dot product
    return abs(X) # Pythagorean theorem'''

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

X = DFT(fx) / N # Divide by N to prevent the amplitude from being too big(Normalization)

plt.stem(frequency_domain, X, 'b', markerfmt = ' ', basefmt = 'b')
#plt.plot(frequency_domain, X)
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.show()
~~~
## Output ( f(x) = 3sin(2t) + sin(7.4t) )
![image](https://user-images.githubusercontent.com/67142421/155623469-4580c00a-e210-456b-8c3a-e5b0a6358265.png)

# Fast Fourier Transform
> The Discrete Fourier Transform takes **O(n^2)** time because it has a nested loop, that is, it is slow.
> The Fast Fourier transform was made to solve this problem and is a essential core of the modern technology.

![image](https://user-images.githubusercontent.com/67142421/155605699-0773c7d0-99fa-4773-ac15-3ddf48958146.png)

~~~Python
~~~

## Output
