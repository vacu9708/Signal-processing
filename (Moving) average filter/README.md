# Average filter
>The average filter is the simplest filter used to remove noise of a signal, but requires more time than the Moving average filter(explained below) for the same accuracy.
>It removes the noise by taking the average at a point of the signal for an instant.

## Code (The original signal is a noisy sinusoidal wave)
~~~Python
import math
import random
from matplotlib import pyplot

def noise():   
    return random.randint(-20, 20) * 0.01
sensor_values = [] # For plot
filtered_sensor_values = [] # For plot
for time in range(333):
    # Finding the average
    filtered_sensor_value = 0
    for i in range(111):
        filtered_sensor_value += math.sin(time * 0.1) + noise() # analogRead
    filtered_sensor_value /= 111
    #-----
    # For plot
    sensor_value = math.sin(time * 0.1) + noise()
    sensor_values.append(sensor_value)
    filtered_sensor_values.append(filtered_sensor_value)
    #-----
# Print
pyplot.plot(sensor_values)
pyplot.plot(filtered_sensor_values)
pyplot.show()
~~~

## Output (Blue : original signal, Orange : filtered signal)
![image](https://user-images.githubusercontent.com/67142421/153942671-38bbdeab-1726-48bc-b8c1-7385ff377a75.png)


# Moving average filter
>The moving average filter is the most common filter used to remove noise in Digital Signal processing, mainly because it is the easiest to understand and use as well as
>the fastest.<br>
>The signal by Moving average filter is delayed quite a bit because previous values are used for smoothing.

## Working process
* It smoothes a noisy signal(removes the noise) by taking the average of a subset of the given series that includes previous values.
* Then the subset is modified by "shifting forward"; that is, excluding the first number of the series(the oldest value) and including the next value in the subset.

## Code (The original signal is a noisy sinusoidal wave)
~~~Python
import math
import random
from matplotlib import pyplot

sensor_values_subset = [] # Will be a sinusoidal wave.
# Initialization
for i in range(20):
    sensor_values_subset.append(0)
#-----
sensor_values = [] # For plot
filtered_sensor_values = [] # For plot
for time in range(333):
    # Removing the oldest value and putting the next value
    sensor_values_subset.pop(0)
    noise = random.randint(-20, 20) * 0.01
    sensor_values_subset.append(math.sin(time * 0.1) + noise) # analogRead
    #-----
    # Finding the average
    filtered_sensor_value = 0
    for sensor_value in sensor_values_subset:
        filtered_sensor_value += sensor_value
    filtered_sensor_value /= 20
    #-----
    #For plot
    sensor_values.append(sensor_values_subset[19])
    filtered_sensor_values.append(filtered_sensor_value)
    #-----
# Print
pyplot.plot(sensor_values)
pyplot.plot(filtered_sensor_values)
pyplot.show()
~~~

## Output (Blue : original signal, Orange : filtered signal)
![image](https://user-images.githubusercontent.com/67142421/153942769-f818e0c7-2621-4b09-b7c4-96c2373bb3d1.png)
