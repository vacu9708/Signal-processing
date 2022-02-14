# Median filter
>The filters where averages are used can hardly deal with impulse noises(spikes in a signal). Median filter is needed for this case.<br>
>It is usually used for image processing.

~~~Python
import math
import random
from matplotlib import pyplot

def noise():
    # An impulse noise by 7% probability, else, a normal noise
    if random.randint(0, 100) <= 7:
        return 1
    else:
        return random.randint(-10, 10) * 0.01
    #-----

values_subset = []
for i in range(20):
    values_subset.append(0)

values = [] # For plot
filtered_values = [] # For plot
for time in range(333):
    # Finding the median of a subset
    value = 0
    for i in range(20):
        value = math.sin(time * 0.1) + noise() # same as analogRead
        values_subset[i] = value
    values_subset.sort() # Sort to find the median value
    filtered_value = values_subset[10] # Median value
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
## Output
![image](https://user-images.githubusercontent.com/67142421/153947334-5ce7624c-8ffe-4042-8961-a72212ac2087.png)
