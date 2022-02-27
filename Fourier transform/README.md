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
* maximum frequency limit = sampling frequency / (2 * sampling time)
* The longer time the signal is measured, the better the frequency resolution is. 
  For example : to measure 1 Hz, the signal has to be recorded for 1 second and to measure 0.1 Hz, the signal has to be recorded for 10 seconds.
* The sampling of a signal whose frequencies are not an integer multiple of the frequency resolution results in a jump in the time signal, and a "smeared" FFT spectrum.

~~~Python
import time
from matplotlib import pyplot as plt
import numpy as np
import math
from cmath import exp

start_time = time.time()

pi = np.pi
# Sinusoidal waves
sampling_time = 10
sampling_frequency = 2048
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
frequency_resolution = 0.1
max_frequency = sampling_frequency / (2 * sampling_time)
frequency_domain = np.arange(0, 200, frequency_resolution)
length_of_frequency_domain = len(frequency_domain)
#-----

def DFT2(fx): # Using matrix
    N = len(fx)
    n = np.arange(N)
    k = frequency_domain.reshape((length_of_frequency_domain, 1))
    e = np.exp(-2j * pi * k * n / N)
    fx = fx.reshape((N, 1))
    X = np.dot(e, fx) # Dot product
    return X # Pythagorean theorem'''

def DFT(fx):
    N = len(fx) # Number of sampling
    X_real = np.zeros(length_of_frequency_domain)
    X_imaginary = np.zeros(length_of_frequency_domain)
    X = np.zeros(length_of_frequency_domain) # Fourier transformed function

    for k in range(length_of_frequency_domain):
        for n in range(N):
            X_real[k] += fx[n] * math.cos(2 * pi * sampling_time * frequency_resolution * k * n / N)
            X_imaginary[k] += fx[n] * -math.sin(2 * pi * sampling_time * frequency_resolution * k * n / N)
        X[k] = math.sqrt(X_real[k]**2 + X_imaginary[k]**2) # Pythagorean theorem (|X|)

    return X / (N / 2) # # Divide X by N to prevent the amplitude from being too big(Normalization)

X = DFT(fx)
print('Elapsed time : ',time.time() - start_time)

#plt.plot(frequency_domain, X)
plt.stem(frequency_domain, abs(X), 'b', markerfmt=" ", basefmt="-b")
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.show()
~~~

# Fast Fourier Transform
> The Discrete Fourier Transform takes **O(n^2)** time because it has a nested loop, that is, it is slow.
> The Fast Fourier transform was made to solve this problem and is a essential core of the modern technology.

![image](https://user-images.githubusercontent.com/67142421/155605699-0773c7d0-99fa-4773-ac15-3ddf48958146.png)

~~~Python
import time
from matplotlib import pyplot as plt
import numpy as np
import math

start_time = time.time()

pi = np.pi
# Sinusoidal waves
sampling_time = 1
sampling_frequency = 2048
t = np.arange(0, sampling_time, sampling_time/sampling_frequency) # The longer period the signal is measured, the better the frequency resolution is.
freq = 20
fx = np.sin(2*pi*freq*t)
freq = 40
fx += 2*np.sin(2*pi*freq*t)

plt.plot(t, fx)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.figure()
#-----
# Frequency domain
frequency_resolution = 1
max_frequency = sampling_frequency / (2 * sampling_time)
frequency_domain = np.arange(0, max_frequency, frequency_resolution)
length_of_frequency_domain = len(frequency_domain)
#-----

class Complex_number:
    def __init__(self, real, imaginary):
        self.real = real
        self.imaginary = imaginary
    
    def __add__(self, b):
        return Complex_number(self.real + b.real, self.imaginary + b.imaginary)
    
    def __sub__(self, b):
        return Complex_number(self.real - b.real, self.imaginary - b.imaginary)
    
    def __mul__(self, b):
        return Complex_number(self.real * b.real - self.imaginary * b.imaginary, self.real * b.imaginary + self.imaginary * b.real)

def absolute_complex_array(complex_array):
    result = np.zeros(len(complex_array))
    for i in range(len(complex_array)):
        result[i] = math.sqrt(complex_array[i].real**2 + complex_array[i].imaginary**2)
    return result

def FFT(fx):
    N = len(fx) # N has to be a power of 2 for FFT.

    if N == 1:
        return np.array([Complex_number(fx[0], 0)])
    
    X_even = FFT(fx[::2]) # FFT of the signal at even indices
    X_odd = FFT(fx[1::2]) # at odd indices

    e = np.array([Complex_number for i in range(N//2)])
    for n in range(N//2):
        e[n] = Complex_number(math.cos(2*pi*n / N), math.sin(2*pi*n / N)) * X_odd[n]

    X_left = np.array([Complex_number for i in range(N//2)])
    X_right = np.array([Complex_number for i in range(N//2)])
    for n in range(N//2):
        X_left[n] = X_even[n] + e[n]
        X_right[n] = X_even[n] - e[n]
    X = np.concatenate((X_left, X_right))

    return X

def FFT2(fx):
    N = len(fx) # N has to be a power of 2 for FFT.

    if N == 1:
        return fx
    
    X_even = FFT2(fx[::2]) # FFT of the signal at even indices
    X_odd = FFT2(fx[1::2]) # at odd indices

    e = np.exp(-2j*pi*np.arange(N) / N)
    X = np.concatenate( (X_even + e[:N//2] * X_odd, X_even + e[N//2:] * X_odd) )
    
    #e = np.exp(-2j*pi*np.arange(N//2) / N)
    #X = np.concatenate( (X_even + e * X_odd, X_even - e * X_odd) )

    return X

X = absolute_complex_array(FFT(fx))
#X = abs(FFT2(fx))
print('Elapsed time : ',time.time() - start_time)

one_side = len(fx)//2
frequency_domain = frequency_domain[:one_side]
X = X[:one_side]/one_side
#plt.plot(frequency_domain, X)
plt.stem(frequency_domain, X, 'b', markerfmt=" ", basefmt="-b")
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.show()
~~~

## Output ( f(x) = sin(2t) + 2sin(4.5t) )
![image](https://user-images.githubusercontent.com/67142421/155848726-c0dc0b03-fedb-4295-9f6d-0d60ef41438d.png)
![image](https://user-images.githubusercontent.com/67142421/155848706-20983ffc-9f2b-4412-94db-524cad96c3d1.png)
