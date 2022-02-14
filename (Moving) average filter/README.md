# Average filter
>The average filter is the simplest filter used to remove noise of a signal, but requires more time than the Moving average filter(explained below) for the same accuracy.
>It removes the noise by taking the average at a point of the signal for an instant.

## Code (The original signal is a noisy sinusoidal wave)
~~~Python
import math
import random
from matplotlib import pyplot

sensor_values = [] # Will be sinusoidal
Sensor_values = [] # For plot
filtered_sensor_values = [] # For plot
for time in range(333):
    # Finding the average
    filtered_sensor_value = 0
    for i in range(111):
        noise = random.randint(-1, 1) * 0.1
        filtered_sensor_value += math.sin(time * 0.1) + noise # analogRead
    filtered_sensor_value /= 111
    #-----
    # For plot
    noise = random.randint(-1, 1) * 0.1
    Sensor_value = math.sin(time * 0.1) + noise
    Sensor_values.append(Sensor_value)
    filtered_sensor_values.append(filtered_sensor_value)
    #-----
# Print
pyplot.plot(Sensor_values)
pyplot.plot(filtered_sensor_values)
pyplot.show()
~~~

## Output (Blue : original signal, Orange : filtered signal)
![image](https://user-images.githubusercontent.com/67142421/153919200-7357fca3-f21b-4604-9abc-b13592d3a239.png)


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

sensor_values = [] # Will be a sinusoidal wave.
# Initialization
for i in range(20):
    sensor_values.append(0)
#-----
Sensor_values = [] # For plot
filtered_sensor_values = [] # For plot
for time in range(333):
    # Removing the oldest value and putting the next value
    sensor_values.pop(0)
    noise = random.randint(-1, 1) * 0.1
    sensor_values.append(math.sin(time * 0.1) + noise) # analogRead
    #-----
    # Finding the average
    filtered_sensor_value = 0
    for sensor_value in sensor_values:
        filtered_sensor_value += sensor_value
    filtered_sensor_value /= 20
    #-----
    #For plot
    Sensor_values.append(sensor_values[19])
    filtered_sensor_values.append(filtered_sensor_value)
    #-----
# Print
pyplot.plot(Sensor_values)
pyplot.plot(filtered_sensor_values)
pyplot.show()
~~~

## Output (Blue : original signal, Orange : filtered signal)
![image](https://user-images.githubusercontent.com/67142421/153897055-c0b60f27-6aea-4e0c-80ea-3a636bd3747a.png)
