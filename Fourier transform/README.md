# Fourier transform
![image](https://user-images.githubusercontent.com/67142421/155687402-a9ae5d4a-9baa-4a83-ac6e-b504ebf805df.png)
>A Fourier transform is a method to find the frequencies of a function. The time domain is transformed to a frequency domain.<br>
>The amplitude of a frequency is a complex number, so it needs to be converted to a real number through Pythagorean theorem

## Fourier transform (ω = 2πf)
![image](https://user-images.githubusercontent.com/67142421/155603554-7edd2873-0942-4465-a931-b6f07a5494da.png)

## Discrete time Fourier Transform (Derived by Riemann sum integral from the Fourier Transform in continuous time)
A discrete time Fourier transform is performed to analyze a signal in the frequency domain with a computer.

![image](https://user-images.githubusercontent.com/67142421/155689010-f04e9a51-ccba-4951-81d2-6346de16f5fc.png)
![image](https://user-images.githubusercontent.com/67142421/155687366-75207445-8ab9-49fe-9505-6c11786e877f.png)<br>
* The infinitesimal dt is 1 because it is discrete time.
* n(sample) corresponds to t(time).
* k corresponds to f (k'th frequency in the frequency domain)

### By Euler's formula :
![image](https://user-images.githubusercontent.com/67142421/155604064-dac589d7-b367-4648-9202-df41ea56f8be.png)

### Characteristics
* Maximum frequency limit = sampling frequency / 2
* Frequency resolution 
  >Frequency resolution = sampling frequency / sample buffer size<br>
  >The frequency resolution can be increased by either reducing the sampling frequency or increasing the size of the sample buffer, which means
  >it will take longer to fill the buffer if we desire increased resolution.
* The sampling of a signal whose frequencies are not an integer multiple of the frequency resolution results in a jump in the time signal, and a "smeared" FFT spectrum.

~~~Python
import time
from matplotlib import pyplot as plt
import numpy as np
import math

start_time = time.time()

pi = np.pi

# Sampling
sample_buffer_size = 2**11
sampling_frequency = (2**11)*0.5 # 0.5Hz of frequency resolution. This will take 2 seconds to fill the sample buffer.
fx = np.zeros(sample_buffer_size)

signal_frequency = [2.5, 4.5]
for n in range(sample_buffer_size):
    fx[n] = math.sin(2*pi*signal_frequency[0]*(n/sampling_frequency)) + 2*math.sin(2*pi*signal_frequency[1]*(n/sampling_frequency))
#-----
plt.plot(np.arange(sample_buffer_size), fx)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.figure()
#-----
# Frequency domain
frequency_resolution = sampling_frequency/sample_buffer_size
max_frequency = sampling_frequency / 2
frequency_domain = np.arange(0, max_frequency, frequency_resolution)
max_k = int(max_frequency / frequency_resolution)
#-----

def DFT2(fx): # Using matrix
    N = len(fx)
    n = np.arange(N)
    k = frequency_domain.reshape((max_k, 1))
    e = np.exp(-2j * pi * k * n / N)
    fx = fx.reshape((N, 1))
    X = np.dot(e, fx) # Dot product
    return abs(X) # Pythagorean theorem'''

def DFT(fx):
    N = len(fx) # Number of sampling
    X = np.zeros(max_k) # Fourier transformed function
    X_real = np.zeros(max_k)
    X_imaginary = np.zeros(max_k)

    for k in range(max_k):
        for n in range(N):
            X_real[k] += fx[n] * math.cos(2 * pi * k * n / N)
            X_imaginary[k] += fx[n] * -math.sin(2 * pi * k * n / N)
        X[k] = math.sqrt(X_real[k]**2 + X_imaginary[k]**2) # Pythagorean theorem (|X|)

    return X / (N / 2) # # Divide X by N to prevent the amplitude from being too big(Normalization)

X = DFT(fx)

print('Elapsed time : ',time.time() - start_time)
#plt.plot(frequency_domain, X)
plt.stem(frequency_domain, X, 'b', markerfmt=" ", basefmt="-b")
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.show()
~~~

# Fast Fourier Transform
> The Discrete Fourier Transform takes **O(n^2)** time because it has a nested loop, that is, it is slow.
> The Fast Fourier transform was made to solve this problem and is an essential core of the signal processing.

## Speed difference between DFT and FFT
![image](https://user-images.githubusercontent.com/67142421/155605699-0773c7d0-99fa-4773-ac15-3ddf48958146.png)

## How to derive FFT
![image](https://user-images.githubusercontent.com/67142421/158342808-af6c272c-cce6-41de-999a-8af8bb85acfd.png)
### Therefore,
![image](https://user-images.githubusercontent.com/67142421/155988816-faf0e483-79bf-4088-b289-80370effb376.png)

![image](https://user-images.githubusercontent.com/67142421/158124937-f4da4cc6-8eb6-4d17-ba14-9c60cb65790e.png)

~~~Python
import time
from matplotlib import pyplot as plt
import numpy as np
import math

start_time = time.time()

pi = np.pi

# Sampling
sample_buffer_size = 2**11
sampling_frequency = (2**11)*0.5 # 0.5Hz of frequency resolution. This will take 2 seconds to fill the sample buffer.
fx = np.zeros(sample_buffer_size)

signal_frequency = [2.5, 4.5]
for n in range(sample_buffer_size):
    fx[n] = math.sin(2*pi*signal_frequency[0]*(n/sampling_frequency)) + 2*math.sin(2*pi*signal_frequency[1]*(n/sampling_frequency))
#-----
plt.plot(np.arange(sample_buffer_size), fx)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.figure()
#-----
# Frequency domain
frequency_resolution = sampling_frequency/sample_buffer_size
max_frequency = sampling_frequency / 2
frequency_domain = np.arange(0, max_frequency, frequency_resolution)
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

def absolute_complex_array(complex_array): # Converting complex numbers to real numbers through Pythagorean theorem
    result = np.zeros(len(complex_array))
    for i in range(len(complex_array)):
        result[i] = math.sqrt(complex_array[i].real**2 + complex_array[i].imaginary**2)
    return result

def FFT(fx):
    N = len(fx) # N has to be a power of 2 for FFT.

    if N == 1:
        return np.array([Complex_number(fx[0], 0)])
    
    X_even = FFT(fx[::2]) # Fourier transformed function of the signal at even indices
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

X = absolute_complex_array(FFT(fx)) / (len(fx)/2)
#X = abs(FFT2(fx))
X = X[:len(fx)//2]

print('Elapsed time : ',time.time() - start_time)
#plt.plot(frequency_domain, X)
plt.stem(frequency_domain, X, 'b', markerfmt=" ", basefmt="-b")
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.show()
~~~

## Output ( f(x) = sin(2t) + 2sin(4.5t) )
![image](https://user-images.githubusercontent.com/67142421/155848726-c0dc0b03-fedb-4295-9f6d-0d60ef41438d.png)
![image](https://user-images.githubusercontent.com/67142421/155848706-20983ffc-9f2b-4412-94db-524cad96c3d1.png)
