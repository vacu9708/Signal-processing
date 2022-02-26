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
import time
from matplotlib import pyplot as plt
import numpy as np
import math

start_time = time.time()

pi = np.pi
# Sinusoidal waves
sampling_time = 10
sampling_frequency = 2048*2
t = np.arange(0, sampling_time, sampling_time/sampling_frequency) # The longer period the signal is measured, the better the frequency resolution is.
freq = 2
fx = np.sin(2*pi*freq*t)
freq = 4.5
fx += 2*np.sin(2*pi*freq*t)

plt.plot(t, fx)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.figure()
#-----
# Frequency domain
max_frequency = sampling_frequency / 2
frequency_resolution = 0.1
frequency_domain = np.arange(0, 200, frequency_resolution)
length_of_frequency_domain = len(frequency_domain)
#-----

'''def DFT2(fx): # Using matrix
    N = len(fx)
    n = np.arange(N)
    k = frequency_domain.reshape((length_of_frequency_domain, 1))
    e = np.exp(-2j * pi * k * n / N)
    fx = fx.reshape((N, 1))
    X = np.dot(e, fx) # Dot product
    return X / (N / 2) # Pythagorean theorem'''

def DFT(fx):
    N = len(fx) # Number of sampling
    X_real = np.zeros(length_of_frequency_domain)
    X_imaginary = np.zeros(length_of_frequency_domain)
    X = np.zeros(length_of_frequency_domain) # Fourier transformed function

    for k in range(int(length_of_frequency_domain)):
        for n in range(N):
            X_real[k] += fx[n] * math.cos(2 * pi * sampling_time * frequency_resolution * k * n / N)
            X_imaginary[k] += fx[n] * math.sin(2 * pi * sampling_time * frequency_resolution * k * n / N)
        X[k] = math.sqrt(X_real[k]**2 + X_imaginary[k]**2) # Pythagorean theorem

    return X / (N / 2) # # Divide X by N to prevent the amplitude from being too big(Normalization)

def FFT2(fx):
    N = len(fx)
    
    if N == 1:
        return fx
    elif N % 2 != 0:
        print('DFT is performed instead since N has to be a power of 2 for FFT')
        return DFT(fx)
    else:
        X_even = FFT2(fx[::2])
        X_odd = FFT2(fx[1::2])
        factor = np.exp(-2j*pi*np.arange(N) / N)
        
        X = np.concatenate( [X_even+factor[:int(N/2)]*X_odd, X_even+factor[int(N/2):]*X_odd] )
        return X

X = DFT(fx)
print('Elapsed time : ',time.time() - start_time)

'''n_oneside = len(fx)//2
frequency_domain = frequency_domain[:n_oneside]
X = X[:n_oneside]/n_oneside'''
plt.plot(frequency_domain, abs(X))
#plt.stem(frequency_domain, abs(X), 'b', markerfmt=" ", basefmt="-b")
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.show()
~~~
## Output ( f(x) = sin(2t) + 2sin(4.5t) )
![image](https://user-images.githubusercontent.com/67142421/155848552-e68560a6-353b-427a-b1fe-fa2e2fb31071.png)
![image](https://user-images.githubusercontent.com/67142421/155848706-20983ffc-9f2b-4412-94db-524cad96c3d1.png)

# Fast Fourier Transform
> The Discrete Fourier Transform takes **O(n^2)** time because it has a nested loop, that is, it is slow.
> The Fast Fourier transform was made to solve this problem and is a essential core of the modern technology.

![image](https://user-images.githubusercontent.com/67142421/155605699-0773c7d0-99fa-4773-ac15-3ddf48958146.png)

~~~Python
~~~

## Output
