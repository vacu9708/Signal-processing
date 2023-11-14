![image](https://github.com/vacu9708/Signal-processing/assets/67142421/ef72ef72-ffce-41cd-bb5f-4d54bb0470a4)<meta name="google-site-verification" content="nrOobyUoslWeNsafQbneSXSrFu-aU7ckZ-LwiT7ChVY" />

# Fourier transform
![image](https://user-images.githubusercontent.com/67142421/155687402-a9ae5d4a-9baa-4a83-ac6e-b504ebf805df.png)
>A Fourier transform is a method to find the frequencies of a signal. The time domain is transformed to a frequency domain.<br>
>The Fourier transform makes a frequency domain expressed as complex numbers, so they need to be converted to real numbers to see the real values.

# How to derive the Fourier transform
## [The fourier series is derived here](https://github.com/vacu9708/Signal-processing/tree/main/Fourier%20series)
T = a period, t = time

![image](https://github.com/vacu9708/Signal-processing/assets/67142421/37877be5-4750-4f8e-8624-3f663cd69c9a)

1. We know that (1/T) is a frequency. When T apporaches infinity, (1/T) becomes an infinitesimal frequency.<br>
2. Let infinitesimal frequency **(1/T) be dk** and n'th frequency **(n/T) be k**.<br>
3. Let us suppose that the domain range is from -infinity to +infinity

![image](https://user-images.githubusercontent.com/67142421/158805467-fbd0db34-250a-43fa-8201-ed9b040fcc8b.png)<br>
![image](https://user-images.githubusercontent.com/67142421/155689010-f04e9a51-ccba-4951-81d2-6346de16f5fc.png)<br>
The outside Riemann sum of expression **(a)** can be converted to integral.<br>

## The derived result : Fourier Transform and Inverse Fourier Transform
![image](https://github.com/vacu9708/Signal-processing/assets/67142421/b50e468e-b125-4a2c-a703-177beced400e)<br>
This expression means:
- The red integral is in frequency domain because integration with respect to time eliminates time. **Fourier Transform**
- The same way, the green integral converts the frequency domain to time domain, that is, the original function. **Inverse Fourier Transform**

## Fourier transform (Angular frequency ω = 2πk)
![image](https://user-images.githubusercontent.com/67142421/155603554-7edd2873-0942-4465-a931-b6f07a5494da.png)

## Discrete time Fourier Transform
A discrete time Fourier transform is performed to analyze a signal in the frequency domain.

![image](https://user-images.githubusercontent.com/67142421/158687851-e2ff15c5-a65a-4e61-8a31-a5d3585f9b2c.png)<br>
- The last index is N-1 because the first index is 0.
- The infinitesimal dt is 1 because it is discrete time.
- n(sample) corresponds to t(time).
- k is k'th frequency in the frequency domain

### By Euler's formula :
![image](https://user-images.githubusercontent.com/67142421/155604064-dac589d7-b367-4648-9202-df41ea56f8be.png)

### Characteristics
* Maximum frequency limit = sampling frequency / 2 (Nyquist frequency)
* Frequency resolution = 1 / recording time (e.g. : Recording 2 seconds gives 0.5Hz resolution, recording 10 seconds gives 0.1Hz resolution.)
* The sampling of a signal whose frequencies are not an integer multiple of the frequency resolution results in a jump in the time signal, and a "smeared" FFT spectrum.
* The values on the right side of the niquist frequency are mirror frequencies, which have the opposite sign in the imaginary number.

### Frequency shift(I noted this down without being sure)
Frequency Shift, or Modulation
* Multiplying exp(-j2πf0t) in the time domain is the same as shifting f0. >> exp(j2πf0t) x(t) = X(f-f0)
* cos(2πf0t) x(t) = 1/2 [ X(f-f0) + X(f+f0) ]
* sin(2πf0t) x(t) =  1/2j [ X(f-f0) - X(f+f0) ]

## Implementing the Discrete Fourier Transform in Python (The output is at the very bottom.)
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
        square_wave[n] += coefficient * math.sin( 2*np.pi*(2*i+1)*(n/sampling_frequency) )'''
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
![image](https://github.com/vacu9708/Signal-processing/assets/67142421/dd0da79c-7e5f-4b33-89cd-11771e2f314e)

### Therefore : ![image](https://user-images.githubusercontent.com/67142421/155988816-faf0e483-79bf-4088-b289-80370effb376.png)<br>
X_even means a fourier-transformed function with regard to the even time points.<br>
The divide and conquer method is applied to this result formula for `NlogN`.<br>

### [Click -> The time complexity of Divide and conquer method](https://github.com/vacu9708/Algorithm/tree/main/Sorting%20algorithm/Merge%20sort)

~~~Python
import time
from matplotlib import pyplot as plt
import numpy as np
import math
import wave#, pyaudio
import random

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

    def __truediv__(self, b):
        return Complex_number(self.real / b, self.imaginary / b)
    '''def show(self):
        print(self.real, self.imaginary, math.sqrt(self.real**2+self.imaginary**2))'''

# Sampling
raw_sound = wave.open('Sound/A string.wav', 'r')
sound = raw_sound.readframes(-1)
sound = np.frombuffer(sound, dtype=int)

sampling_frequency = 48000 # The default sampling frequency is 48000Hz. Decrease it for faster speed
sample_buffer_size = 2**17
signal = np.zeros(sample_buffer_size)

for n in range(sample_buffer_size):
    signal[n] = sound[n * (raw_sound.getframerate()//sampling_frequency)]
#-----
# Frequency domain
frequency_resolution = sampling_frequency/sample_buffer_size
max_frequency = sampling_frequency #/ 2
frequency_domain = np.arange(0, max_frequency, frequency_resolution)
#-----

def absolute_IFFT(complex_array): # Converting complex numbers to real numbers through Pythagorean theorem
    result = np.zeros(len(complex_array))
    for i in range(len(complex_array)):
        sign = -1 if complex_array[i].real + complex_array[i].imaginary < 0 else 1
        result[i] = math.sqrt(complex_array[i].real**2 + complex_array[i].imaginary**2) * sign
    return result

def absolute_FFT(complex_array): # Converting complex numbers to real numbers through Pythagorean theorem
    result = np.zeros(len(complex_array))
    for i in range(len(complex_array)):
        result[i] = math.sqrt(complex_array[i].real**2 + complex_array[i].imaginary**2)
    return result

def FFT(signal):
    N = len(signal) # N has to be a power of 2 for FFT.

    if N == 1: # The fourier transform of a function whose size is 1 makes the original signal.
        return np.array([Complex_number(signal[0], 0)])
    
    X_even = FFT(signal[::2]) # Fourier transformed function of the signal at even indices
    X_odd = FFT(signal[1::2]) # at odd indices

    def e(k):
        return Complex_number(math.cos(2*pi*k / N), -math.sin(2*pi*k / N))

    X = np.array([Complex_number for i in range(N)])
    for k in range(N//2):
        X[k] = X_even[k] + X_odd[k] * e(k)
        X[N//2 + k] = X_even[k] + X_odd[k] * e(N//2 + k) # X[N/2 + k] is equal to X[k] because N/2 is one period

    return X

def inverse_FFT(signal): # Inverse fourier transform of a signal that have become absolute values makes a wrong output.
    N = len(signal)

    if N == 1:
        return signal
    
    X_even = inverse_FFT(signal[::2])
    X_odd = inverse_FFT(signal[1::2])

    def e(k):
        return Complex_number(math.cos(2*pi*k / N), math.sin(2*pi*k / N)) # The minus that was on the sin changes to plus

    X = np.array([Complex_number for i in range(N)])
    for k in range(N//2):
        X[k] = X_even[k] + X_odd[k] * e(k)
        X[N//2 + k] = X_even[k] + X_odd[k] * e(N//2 + k) 

    return X

def find_main_frequency(X, frequency_resolution):
    '''Using the local max point
    for i in range(20, sample_buffer_size//2):
        if X[i] < X[i+1] and X[i+1] > X[i+2]: # If a local max has been found
            # Find the average of the previous values
            sum = 0
            for j in range(5, 10):
                sum += X[i-j]
            average = sum / 5
            #-----
            if X[i+1] > average*10: # If the local max is much bigger than the average of the previous values
                i_local_max = i+1 # Determine the index of the main frequency
                break
    #-----'''
    # Find the frequency with the biggest amplitude
    index_max = 0
    for i in range(sample_buffer_size//2):
        if X[i] > X[index_max]:
            freq_with_max_amplitude = i
    # Proportional distribution to mitigate the frequency error value
    proportional_distribution = 1 / (X[freq_with_max_amplitude-1]+X[freq_with_max_amplitude]+X[freq_with_max_amplitude+1])
    frequency1 = X[freq_with_max_amplitude-1] * proportional_distribution * (freq_with_max_amplitude-1)
    frequency2 = X[freq_with_max_amplitude] * proportional_distribution * freq_with_max_amplitude
    frequency3 = X[freq_with_max_amplitude+1] * proportional_distribution * (freq_with_max_amplitude+1)
    main_frequency = (frequency1+frequency2+frequency3)*frequency_resolution
    #-----
    return main_frequency

X = FFT(signal) / (sample_buffer_size)
absolute_X = absolute_FFT(X)

main_frequency = find_main_frequency(absolute_X, frequency_resolution)
print('\n\nMain frequency : {}'.format(main_frequency))

inverse_X = absolute_IFFT(inverse_FFT(X))

# Play the sound
# py_audio = pyaudio.PyAudio()
# stream = py_audio.open(output=True,
#             channels=1,
#             rate=int(sampling_frequency),
#             format=pyaudio.paInt32,
#             )
# stream.write(inverse_X.astype(np.int32))
#-----

#-----Plot
plt.title('Sampled signal')
plt.plot(np.arange(0, sample_buffer_size/sampling_frequency, 1/sampling_frequency), signal)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.figure()

plt.title('Frequency domain')
#plt.plot(frequency_domain, absolute_X[:len(signal)//2])
plt.stem(frequency_domain, absolute_X, 'b', markerfmt=" ", basefmt="-b")
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.figure()

plt.title('Inverse FFT')
plt.plot(np.arange(0, sample_buffer_size/sampling_frequency, 1/sampling_frequency), inverse_X)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()
~~~

## Output ( f(x) = sin(2t) + 2sin(4.5t) )
![image](https://user-images.githubusercontent.com/67142421/155848726-c0dc0b03-fedb-4295-9f6d-0d60ef41438d.png)
![image](https://user-images.githubusercontent.com/67142421/155848706-20983ffc-9f2b-4412-94db-524cad96c3d1.png)
## Output ( 6th E string in the standard guitar tuning)
![image](https://user-images.githubusercontent.com/67142421/159037509-04a0ac4b-a5b2-42b6-aaf2-13b0c1ece52f.png)
## Inverse fourier transform after modifying the signal (After eliminating the main frequency)
![image](https://user-images.githubusercontent.com/67142421/158883908-8c25e482-d883-4425-ab2e-152c8daf9dae.png)

![image](https://user-images.githubusercontent.com/67142421/158881956-2dfe675b-cb48-4ce2-a227-33aa4597511d.png)
