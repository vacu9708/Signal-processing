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

    e = np.array([Complex_number for i in range(N)])
    for k in range(N):
        e[k] = Complex_number(math.cos(2*pi*k / N), -math.sin(2*pi*k / N))

    X = np.array([Complex_number for i in range(N)])
    for k in range(N//2):
        X[k] = X_even[k] + X_odd[k] * e[k]
        X[N//2 + k] = X_even[k] + X_odd[k] * e[N//2 + k] # N/2 + k is equal to k because N/2 is one period in X_even and X_odd.

    return X

def inverse_FFT(signal): # The inverse fourier transform of a signal that have become absolute values makes a wrong output.
    N = len(signal) # N has to be a power of 2 for FFT.

    if N == 1: # The fourier transform of a signal whose size is 0 makes the original signal.
        return signal # Has to be a complex number
    
    X_even = inverse_FFT(signal[::2]) # Fourier transformed function of the signal at even indices
    X_odd = inverse_FFT(signal[1::2]) # at odd indices

    e = np.array([Complex_number for i in range(N)])
    for k in range(N):
        e[k] = Complex_number(math.cos(2*pi*k / N), math.sin(2*pi*k / N)) # The minus that was on the sin changes to plus

    X = np.array([Complex_number for i in range(N)])
    for k in range(N//2):
        X[k] = X_even[k] + X_odd[k] * e[k]
        X[N//2 + k] = X_even[k] + X_odd[k] * e[N//2 + k] # N/2 + k is equal to k because N/2 is one period in X_even and X_odd.

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
            index_max = i
    # Proportional distribution
    proportional_distribution = 1 / (X[index_max-1]+X[index_max]+X[index_max+1])
    frequency1 = X[index_max-1] * proportional_distribution * (index_max-1)
    frequency2 = X[index_max] * proportional_distribution * index_max
    frequency3 = X[index_max+1] * proportional_distribution * (index_max+1)
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