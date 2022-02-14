# Median filter
>Average filter cannot remove impulse noises(spikes in a signal). Median filter is needed for this case.

~~~Python
import math
import random
from matplotlib import pyplot

def noise():
    # An impulse noise by 10% probability, else, a normal noise
    if random.randint(0, 100) <= 10:
        return 1
    else:
        return random.randint(-10, 10) * 0.01
    #-----

sensor_values_subset = []
for i in range(20):
    sensor_values_subset.append(0)

sensor_values = [] # For plot
filtered_sensor_values = [] # For plot
for time in range(333):
    # Finding the median of a subset
    for i in range(20):
        sensor_values_subset[i] = math.sin(time * 0.1) + noise() # analogRead
    sensor_values_subset.sort()
    filtered_sensor_value = sensor_values_subset[10]
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
