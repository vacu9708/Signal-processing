# Fourier transform filter
>Fourier transform and Inverse fourier transform are going to be used to get rid of the noise here.
>The noise will be gotten rid of in the frequency domain, and then be inverse fourier transformed into the time domain.

~~~Python
from matplotlib import pyplot as plt
import numpy as np
import math
import wave, pyaudio
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
raw_sound = wave.open('Sound/Recording.wav', 'r')
sound = raw_sound.readframes(-1)
sound = np.frombuffer(sound, dtype=int)

def noise():   
    return random.randint(-10**8, 10**8)

'''sampling_frequency = 2**14
sample_buffer_size = 2**15
signal = np.zeros(sample_buffer_size)'''

'''for n in range(sample_buffer_size):
    signal[n] = 10**9*math.sin(2*pi*300*(n/sampling_frequency)) + noise() # same as analogRead'''

sampling_frequency = 48000
sample_buffer_size = 2**18
signal = np.zeros(sample_buffer_size)

for n in range(raw_sound.getnframes()):
    signal[n] = sound[n * (raw_sound.getframerate()//sampling_frequency)] + noise()
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

    if N == 1: # The fourier transform of a function whose size is 0 makes the original signal.
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

def frequency_filter(X, max_freq):
    for i in range(int((max_freq+1)/frequency_resolution), int(sample_buffer_size/2)):
        X[i] = Complex_number(0,0)
        X[sample_buffer_size-i] = Complex_number(0,0)

def threshold_filter(X, threshold):
    absolute_X = absolute_FFT(X)
    for i in range(sample_buffer_size):
        if absolute_X[i] < threshold:
            X[i] = Complex_number(0,0)

X = FFT(signal) / (sample_buffer_size)
absolute_X = absolute_FFT(X)
frequency_filter(X, 3*10**3)
#threshold_filter(X, 4*10**5)
modified_X = absolute_FFT(X)

inverse_X = absolute_IFFT(inverse_FFT(X))

# Save the audios
wavs = [wave.open('./Sound/Voice with noise.wav', 'w'), wave.open('./Sound/Voice with noise gotten rid of.wav', 'w')]
wavs[0].setparams(raw_sound.getparams())
wavs[0].writeframes(signal.astype(np.int32).tobytes())
wavs[0].close()
wavs[1].setparams(raw_sound.getparams())
wavs[1].writeframes(inverse_X.astype(np.int32).tobytes())
wavs[1].close()
#-----
# Play the sound
py_audio = pyaudio.PyAudio()
stream = py_audio.open(output=True,
            channels=1,
            rate=int(sampling_frequency),
            format=pyaudio.paInt32,
            )
stream.write(inverse_X.astype(np.int32).tobytes())
#-----

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

plt.title('Modified frequency domain')
plt.stem(frequency_domain, modified_X, 'b', markerfmt=" ", basefmt="-b")
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.figure()

plt.title('Modified signal')
plt.plot(np.arange(0, sample_buffer_size/sampling_frequency, 1/sampling_frequency), inverse_X)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()
~~~
## Output
![image](https://user-images.githubusercontent.com/67142421/159341958-3c22c8a6-9218-493b-b1f5-0259a2fc774e.png)
![image](https://user-images.githubusercontent.com/67142421/159342186-2e28b3c7-bdd8-460b-be15-fb5482cdc77f.png)
![image](https://user-images.githubusercontent.com/67142421/159342211-e15f7a74-1cd9-4c9b-8c3d-88a7a738059f.png)
![image](https://user-images.githubusercontent.com/67142421/159342267-73c93b0e-aa52-4fb2-aebe-b6f04ef12e33.png)




