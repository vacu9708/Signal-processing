# How to derive fourier series
![Piror knowledge](https://user-images.githubusercontent.com/67142421/154923818-be9592f1-b4aa-4b9d-b68b-a046b388e1fb.jpg)
![Another way when n=m](https://user-images.githubusercontent.com/67142421/154923847-9f294c3f-98b1-4e8c-9074-858640b37ede.jpg)
![Derivation](https://user-images.githubusercontent.com/67142421/154923859-250a665b-b8da-4e3b-a68a-c2278874d83b.jpg)

# Making a square wave

## Deriving the fourier series of a square wave
![Deriving the fourier series of a square wave](https://user-images.githubusercontent.com/67142421/154939586-b14b9984-4fcd-4efc-a0a0-77ec0d4f5336.jpg)

## Graph
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

fourier_series = []
for x in range(-333, 333):
    fourier_series.append(fx(x * 0.1, 2))

x = numpy.array(range(-333, 333))
pyplot.plot(x, fourier_series)
#pyplot.legend()
pyplot.show()
~~~

##Output
![image](https://user-images.githubusercontent.com/67142421/154935742-871c2a93-b759-40b3-9710-778fd68ae1a5.png)
