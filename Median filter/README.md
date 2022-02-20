# Median filter
>The filters where averages are used can hardly remove impulse noises(spikes in a signal) because the average includes the influence of the spike. Median filter is needed for this situation.<br>
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
        value = math.sin((time * 0.1) + (i % 10 * 0.01)) + noise() # same as analogRead
        values_subset[i] = value
    values_subset.sort() # Sort to find the median value
    filtered_value = values_subset[10] # Median value
    #-----
    # For plot
    values.append(value)
    filtered_values.append(filtered_value)
    #-----
# Print
pyplot.plot(values, label = 'Original signal')
pyplot.plot(filtered_values, label = 'Filtered signal')
pyplot.legend()
pyplot.show()
~~~
## Output
![image](https://user-images.githubusercontent.com/67142421/154861725-8d06371d-40f7-4343-a11f-4fbcc4254367.png)
