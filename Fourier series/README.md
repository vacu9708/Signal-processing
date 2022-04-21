![image](https://user-images.githubusercontent.com/67142421/154940640-606a5587-61af-45b3-809f-b1b455b6f237.png)

# Fourier series
>Fourier series starts from the idea that all periodic functions can be expressed as the sum of trigonometric functions.
>Various shapes of function can be made with Fourier series.

![image](https://user-images.githubusercontent.com/67142421/164544510-187ef91a-d535-4abb-a7df-4e965459cb1b.png)

## How to derive fourier series
![Piror knowledge](https://user-images.githubusercontent.com/67142421/154923818-be9592f1-b4aa-4b9d-b68b-a046b388e1fb.jpg)
![Another way when n=m](https://user-images.githubusercontent.com/67142421/154923847-9f294c3f-98b1-4e8c-9074-858640b37ede.jpg)
![Untitled](https://user-images.githubusercontent.com/67142421/164543201-68405160-379d-48c4-9f6e-521be53dd940.png)

## Making a square wave
![Deriving the fourier series of a square wave](https://user-images.githubusercontent.com/67142421/154939586-b14b9984-4fcd-4efc-a0a0-77ec0d4f5336.jpg)

## Complex fourier series
![image](https://user-images.githubusercontent.com/67142421/161392050-86a93cef-f557-4cec-9e96-06959921756b.png)

![KakaoTalk_20220317_220136668](https://user-images.githubusercontent.com/67142421/158814349-aa196cc3-d1d2-4b70-a8e2-563bbf1b57d5.jpg)

## Code
~~~Python
import math
from matplotlib import pyplot
import numpy

def fx(x, k):
    y = 0
    for n in range(1, 9876):
        #y += ( 2 * k / (n * math.pi) - (2 * k / (n * math.pi)) * math.cos(n * math.pi) ) * math.sin(n * x)
        y += ( 4 * k / (math.pi * (2 * n - 1)) ) * math.sin((2 * n - 1) * x)
    return y

fourier_series_of_square_wave = []
for x in range(-333, 333):
    fourier_series_of_square_wave.append(fx(x * 0.1, 2))

x = numpy.array(range(-333, 333))
pyplot.plot(x, fourier_series_of_square_wave)
#pyplot.legend()
pyplot.show()
~~~

## Output
![image](https://user-images.githubusercontent.com/67142421/154935742-871c2a93-b759-40b3-9710-778fd68ae1a5.png)
