import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')

x_vals = []
y_vals = []

index = count()


def animate(i):
    data = pd.read_csv('../realtimeData.csv')
    x = data['x_value']
    y1 = data['config.Human_Data']
    y2 = data['detectedViolators']
    y3 = data['totalViolations']

    plt.cla()

    plt.plot(x, y1, label='Detected Person')
    plt.plot(x, y2, label='Violators')
    plt.plot(x, y3, label='Violations')
    

    plt.legend(loc='upper left')
    plt.title('Data Visualization')
    plt.tight_layout()


ani = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.tight_layout()
plt.show()