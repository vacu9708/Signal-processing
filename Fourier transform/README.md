# Fourier transform
![image](https://user-images.githubusercontent.com/67142421/155687402-a9ae5d4a-9baa-4a83-ac6e-b504ebf805df.png)
>A Fourier transform is a method to find the frequencies of a function. The time domain is transformed to a frequency domain.<br>
>The Fourier transform makes a frequency domain expressed as complex numbers, so they need to be converted to real numbers to see the real value through the Pythagorean theorem.

## [The Fourier transform is derived from the Fourier series](https://github.com/vacu9708/Signal-processing/tree/main/Fourier%20series)
![image](https://user-images.githubusercontent.com/67142421/158811827-513af3b1-0dfb-4319-a7d8-6c2ffa9b5688.png)<br>
>We know that (1/T) is a frequency. When T apporaches infinity, (1/T) becomes an infinitesimal frequency.<br>
> Let infinitesimal frequency **(1/T) be dk** and n'th frequency **(n/T) be k**.

![image](https://user-images.githubusercontent.com/67142421/158805467-fbd0db34-250a-43fa-8201-ed9b040fcc8b.png)

![image](https://user-images.githubusercontent.com/67142421/155689010-f04e9a51-ccba-4951-81d2-6346de16f5fc.png)

![image](https://user-images.githubusercontent.com/67142421/158805992-cf16d1f5-26db-4bb8-9379-aabcd485be32.png)<br>

> Expression (1) can be expressed as series by the Riemann sum.<br>
>This expression means integrating the underlined **function, whose domain is comprised of frequencies, with respect to k(frequency)**.<br>
>That is to say, the underlined expression is the very ***Fourier transform*** that we were looking for.<br>
>Also, we can find out the **Inverse Fourier Transform** in this expression. Doing the Fourier transform of a fourier transformed function again converts the frequency domain to a time domain back.

## Fourier transform (Angular frequency ω = 2πf)
![image](https://user-images.githubusercontent.com/67142421/155603554-7edd2873-0942-4465-a931-b6f07a5494da.png)

## Discrete time Fourier Transform
>A discrete time Fourier transform is performed to analyze a signal in the frequency domain with a computer.

![image](https://user-images.githubusercontent.com/67142421/158687851-e2ff15c5-a65a-4e61-8a31-a5d3585f9b2c.png)<br>
The first index of an array is 0. That's why the last index is N-1.
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
* The values on the right side of the niquist frequency are mirror values, which have the opposite sign, either (+) or (-)

### The type of number on each domain
* Frequency domain after a fourier transform : **Complex numbers**
* Time domain after an inverse fourier transform : **Imaginary numbers with all the real numbers being 0**

~~~Python
import time
from matplotlib import pyplot as plt
import numpy as np
import math

start_time = time.time()

pi = np.pi

def DFT(fx):
    N = len(fx) # Number of sampling
    X = np.zeros(N//2) # Fourier transformed function
    X_real = np.zeros(N//2)
    X_imaginary = np.zeros(N//2)

    for k in range(N//2): # k : frequency domain
        for n in range(N):
            X_real[k] += fx[n] * math.cos(2 * pi * k * n / N)
            X_imaginary[k] += fx[n] * -math.sin(2 * pi * k * n / N)
        X[k] = math.sqrt(X_real[k]**2 + X_imaginary[k]**2) # Pythagorean theorem (|X|)

    return X / N # Divide X by N to prevent the amplitude from being too big (Normalization)

# Sampling
sample_buffer_size = 2**11
sampling_frequency = (2**11)*0.5 # 0.5Hz of frequency resolution. This will take 2 seconds to fill the sample buffer.
fx = np.zeros(sample_buffer_size)

signal_frequency = [2.5, 5]
for n in range(sample_buffer_size):
    # n/sampling_frequency : Time taken per sample
    fx[n] = math.sin(2*pi*signal_frequency[0]*(n/sampling_frequency)) + 2*math.sin(2*pi*signal_frequency[1]*(n/sampling_frequency))
'''Square wave
for n in range(sample_buffer_size):
    for i in range(9999):
        coefficient = 1 / (2*i+1)
        square_wave[n] += coefficient * math.sin( np.pi*(2*i+1)*(n/sampling_frequency) )'''
#-----
# Plot the sampled signal
# sample_buffer_size/sampling_frequency : Total time taken for sampling
plt.plot(np.arange(0, sample_buffer_size/sampling_frequency, 1/sampling_frequency), fx)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.figure()
#-----
# Frequency domain
frequency_resolution = sampling_frequency/sample_buffer_size
max_frequency = sampling_frequency / 2
frequency_domain = np.arange(0, max_frequency, frequency_resolution)
#-----

X = DFT(fx)

#plt.plot(frequency_domain, X)
plt.stem(frequency_domain, X, 'b', markerfmt=" ", basefmt="-b")
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
print('Elapsed time : ',time.time() - start_time)
plt.show()
~~~

# Fast Fourier Transform
> The Discrete Fourier Transform takes **O(n^2)** time because it has a nested loop, that is, it is slow.<br>
> The Fast Fourier transform algorithm was made to solve this problem and is one of the essential cores of signal processing.<br>
> The main idea is that *Divide and conquer* method can be used to convert it to an algorithm that takes **O(NlogN)**.

## Speed difference between DFT and FFT
![image](https://user-images.githubusercontent.com/67142421/155605699-0773c7d0-99fa-4773-ac15-3ddf48958146.png)

## How to derive FFT
![image](https://user-images.githubusercontent.com/67142421/158342808-af6c272c-cce6-41de-999a-8af8bb85acfd.png)
### Therefore : ![image](https://user-images.githubusercontent.com/67142421/155988816-faf0e483-79bf-4088-b289-80370effb376.png)<br>
The divide and conquer method can be applied with this result to increase the speed.

### [Click -> The time complexity of Divide and conquer method](https://github.com/vacu9708/Algorithm/tree/main/Sorting%20algorithm/Merge%20sort)

~~~Python
import time
from matplotlib import pyplot as plt
import numpy as np
import math
import wave

start_time = time.time()

pi = np.pi

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

def get_real(complex_array):
    result = np.zeros(len(complex_array))
    for i in range(len(complex_array)):
        result[i] = complex_array[i].real
    return result

def absolute_complex_array(complex_array): # Converting complex numbers to real numbers through Pythagorean theorem
    result = np.zeros(len(complex_array))
    for i in range(len(complex_array)):
        result[i] = math.sqrt(complex_array[i].real**2 + complex_array[i].imaginary**2)
    return result

def FFT(fx):
    N = len(fx) # N has to be a power of 2 for FFT.

    if N == 1: # The fourier transform of a function whose size is 0 makes the original signal.
        return np.array([Complex_number(fx[0], 0)])
    
    X_even = FFT(fx[::2]) # Fourier transformed function of the signal at even indices
    X_odd = FFT(fx[1::2]) # at odd indices

    e = np.array([Complex_number for i in range(N)])
    for k in range(N):
        e[k] = Complex_number(math.cos(2*pi*k / N), -math.sin(2*pi*k / N))

    X = np.array([Complex_number for i in range(N)])
    for k in range(N//2):
        X[k] = X_even[k] + X_odd[k] * e[k]
        X[N//2 + k] = X_even[k] + X_odd[k] * e[N//2 + k] # N/2 + k is equal to k because N/2 is one period in X_even and X_odd.

    return X

def inverse_FFT(fx): # The inverse fourier transform of a signal that have become absolute values makes a wrong output.
    N = len(fx) # N has to be a power of 2 for FFT.

    if N == 1: # The fourier transform of a signal whose size is 0 makes the original signal.
        return fx # Has to be a complex number
    
    X_even = inverse_FFT(fx[::2]) # Fourier transformed function of the signal at even indices
    X_odd = inverse_FFT(fx[1::2]) # at odd indices

    e = np.array([Complex_number for i in range(N)])
    for k in range(N):
        e[k] = Complex_number(math.cos(2*pi*k / N), math.sin(2*pi*k / N)) # The minus that was on the sin changes to plus

    X = np.array([Complex_number for i in range(N)])
    for k in range(N//2):
        X[k] = X_even[k] + X_odd[k] * e[k]
        X[N//2 + k] = X_even[k] + X_odd[k] * e[N//2 + k] # N/2 + k is equal to k because N/2 is one period in X_even and X_odd.

    return X

def FFT2(fx):
    N = len(fx) # N has to be a power of 2 for FFT.

    if N == 1:
        return fx
    
    X_even = FFT2(fx[::2]) # FFT of the signal at even indices
    X_odd = FFT2(fx[1::2]) # at odd indices

    e = np.exp(-2j*pi*np.arange(N) / N)
    X = np.concatenate( (X_even + X_odd * e[:N//2], X_even + X_odd * e[N//2:]) )

    return X

def find_main_frequency(X, frequency_resolution):
    # Find the max and second max frequency
    index_max = 0
    second_max = 0
    for i in range(1, int(1000/frequency_resolution)):
        if X[i] > X[index_max]:
            second_max = index_max
            index_max = i
    #-----
    i_main_freq = index_max if index_max < second_max else second_max # Find the main frequency

    proportional_distribution = 1 / (X[i_main_freq-1]+X[i_main_freq]+X[i_main_freq+1])
    frequency1 = X[i_main_freq-1] * proportional_distribution * (i_main_freq-1)
    frequency2 = X[i_main_freq] * proportional_distribution * i_main_freq
    frequency3 = X[i_main_freq+1] * proportional_distribution * (i_main_freq+1)
    main_frequency = (frequency1+frequency2+frequency3)*frequency_resolution
    return main_frequency

# Sampling
sound = wave.open('Guitar strings/1st E string.wav', 'r')
signal = sound.readframes(-1)
signal = np.frombuffer(signal, dtype=int)

sampling_frequency = 4800 # The default sampling frequency is 48000Hz. Decrease it for faster speed
sample_buffer_size = 2**11
fx = np.zeros(sample_buffer_size)

for n in range(sample_buffer_size):
    fx[n] = signal[n * (sound.getframerate()//sampling_frequency)]

plt.title('Sampled signal')
plt.plot(np.arange(0, sample_buffer_size/sampling_frequency, 1/sampling_frequency), fx)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.figure()
#-----
# Frequency domain
frequency_resolution = sampling_frequency/sample_buffer_size
max_frequency = sampling_frequency / 2
frequency_domain = np.arange(0, max_frequency, frequency_resolution)
#-----

complex_X = FFT(fx)
X = absolute_complex_array(complex_X) / len(fx)
main_frequency = find_main_frequency(X, frequency_resolution)
print('\n\nMain frequency : {}'.format(main_frequency))

# Plot the frequency domain
plt.title('Frequency domain')
plt.plot(frequency_domain, X[:len(fx)//2])
#plt.stem(frequency_domain, X[:len(fx)//2], 'b', markerfmt=" ", basefmt="-b")
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.figure()
#-----
# Eliminating the main frequency and performing a inverse fourier transform.
# Find the maxfrequency
index_max = 0
for i in range(int(1000/frequency_resolution)):
    if X[i] > X[index_max]:
        index_max = i
#-----
# Eliminating the main frequency
complex_X[index_max-1] = Complex_number(0,0)
complex_X[index_max] = Complex_number(0,0)
complex_X[index_max+1] = Complex_number(0,0)
#-----
inverse_X = get_real(inverse_FFT(complex_X)) / len(fx) # Inverse fourier transform.

plt.title('Modified signal')
plt.plot(np.arange(0, sample_buffer_size/sampling_frequency, 1/sampling_frequency), inverse_X)
plt.xlabel('Time')
plt.ylabel('Amplitude')
print('Elapsed time : ',time.time() - start_time)
plt.show()
~~~

## Output ( f(x) = sin(2t) + 2sin(4.5t) )
![image](https://user-images.githubusercontent.com/67142421/155848726-c0dc0b03-fedb-4295-9f6d-0d60ef41438d.png)
![image](https://user-images.githubusercontent.com/67142421/155848706-20983ffc-9f2b-4412-94db-524cad96c3d1.png)
## Output ( 1st E string in standard guitar tuning)
![image](https://user-images.githubusercontent.com/67142421/159037509-04a0ac4b-a5b2-42b6-aaf2-13b0c1ece52f.png)
## Inverse fourier transform after modifying the signal (After eliminating the main frequency)
![image](https://user-images.githubusercontent.com/67142421/158883908-8c25e482-d883-4425-ab2e-152c8daf9dae.png)

![image](https://user-images.githubusercontent.com/67142421/158881956-2dfe675b-cb48-4ce2-a227-33aa4597511d.png)
