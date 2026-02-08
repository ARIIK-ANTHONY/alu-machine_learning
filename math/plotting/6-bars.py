#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

# your code here
people = ['Farrah', 'Fred', 'Felicia']
fruit_names = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

bottom = np.zeros(3)
for i in range(4):
    plt.bar(people, fruit[i], bottom=bottom, color=colors[i], 
            width=0.5, label=fruit_names[i])
    bottom += fruit[i]

plt.ylabel('Quantity of Fruit', fontsize='small')
plt.title('Number of Fruit per Person', fontsize='small')
plt.ylim(0, 80)
plt.yticks(np.arange(0, 81, 10))
plt.legend(fontsize='small')
plt.show()
