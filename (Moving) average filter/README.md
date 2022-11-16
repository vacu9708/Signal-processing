# Average filter
>The average filter is the simplest filter used to remove noise of a signal, but requires more calculations than Moving average filter for the same accuracy.<br>
>It removes the noise by taking the average at a point of the signal for an instant.

## Code (The original signal is a noisy sinusoidal wave)
~~~Python
import math
import random
from matplotlib import pyplot

values = [] # For plot
filtered_values = [] # For plot
for time in range(333):
    filtered_value = 0
    # Finding the average
    for i in range(20):
        noise=random.randint(-20, 20) * 0.01
        value = math.sin(time * 0.1) + noise
        filtered_value += value
    filtered_value /= 20
    # For plot
    values.append(value)
    filtered_values.append(filtered_value)
# Display
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
import math,random
from collections import deque
from matplotlib import pyplot

values = [] # For plot
filtered_values = [] # For plot
LENGTH=20
noisy_values = deque([0]*LENGTH)
sum = 0
for time in range(333):
    # Remove the oldest value
    sum-=noisy_values.popleft()
    # Put a new value
    noise = random.randint(-20, 20) * 0.01
    noisy_value=math.sin(time * 0.1) + noise
    noisy_values.append(noisy_value)
    sum+=noisy_value
    # Find the average
    filtered_value=sum / LENGTH
    # For displaying
    values.append(noisy_value)
    filtered_values.append(filtered_value)
# Display
pyplot.plot(values)
pyplot.plot(filtered_values)
pyplot.show()
~~~

## Output (Blue : original signal, Orange : filtered signal)
![image](https://user-images.githubusercontent.com/67142421/202138129-de96f212-70cc-4355-86a9-bb9045d94549.png)
