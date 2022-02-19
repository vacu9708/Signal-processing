# Average filter
>The average filter is the simplest filter used to remove noise of a signal, but requires more calculations than Moving average filter for the same accuracy.<br>
>It removes the noise by taking the average at a point of the signal for an instant.

## Code (The original signal is a noisy sinusoidal wave)
~~~Python
import math
import random
from matplotlib import pyplot

def noise():   
    return random.randint(-20, 20) * 0.01
values = [] # For plot
filtered_values = [] # For plot
for time in range(333):
    # Finding the average
    value = 0
    filtered_value = 0
    for i in range(20):
        value = math.sin((time * 0.1) + (i % 10 * 0.01)) + noise() # same as analogRead
        filtered_value += value
    filtered_value /= 20
    #-----
    # For plot
    values.append(value)
    filtered_values.append(filtered_value)
    #-----
# Print
pyplot.plot(values)
pyplot.plot(filtered_values)
pyplot.show()
~~~

## Output (Blue : original signal, Orange : filtered signal)
![image](https://user-images.githubusercontent.com/67142421/154814234-d2d88676-a600-473e-bda3-d2af9ec7c203.png)

# Moving average filter
>The moving average filter is the most common filter used to remove noise in Digital Signal processing, mainly because it is the easiest to understand and use as well as
>the fastest, requiring less calculations than Average filter for the same accuracy.<br>
>The signal by Moving average filter is delayed quite a bit because previous values are used for smoothing.

## Working process
* It smoothes a noisy signal(removes the noise) by taking the average of a subset of the given series that includes previous values.
* Then the subset is modified by "shifting forward"; that is, excluding the first number of the series(the oldest value) and including the next value in the subset.

## Code (The original signal is a noisy sinusoidal wave)
~~~Python
import math
import random
from matplotlib import pyplot

values_subset = [] # Will be a sinusoidal wave.
# Initialization
for i in range(20):
    values_subset.append(0)
#-----
values = [] # For plot
filtered_values = [] # For plot
for time in range(333):
    # Removing the oldest value and putting the next value
    values_subset.pop(0)
    noise = random.randint(-20, 20) * 0.01
    values_subset.append(math.sin(time * 0.1) + noise) # same as analogRead
    #-----
    # Finding the average
    filtered_value = 0
    for value in values_subset:
        filtered_value += value
    filtered_value /= 20
    #-----
    #For plot
    values.append(values_subset[19])
    filtered_values.append(filtered_value)
    #-----
# Print
pyplot.plot(values)
pyplot.plot(filtered_values)
pyplot.show()
~~~

## Output (Blue : original signal, Orange : filtered signal)
![image](https://user-images.githubusercontent.com/67142421/153942769-f818e0c7-2621-4b09-b7c4-96c2373bb3d1.png)
