import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
from pathlib import Path

folder =   Path(r'../data.txt')

df = pd.read_csv(folder, sep=" ", header=None)
df.columns = ["x", "y1", "y2"]
print(df)

fig, ax = plt.subplots()

ax.plot(df['x'],df['y1'], label='Violations')
ax.plot(df['x'],df['y2'], label='Dummy Data')
tick_spacing = 1
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
plt.xlabel('Date')
plt.ylabel('Data')
plt.title('Plot of Data')
plt.legend()
plt.show()
