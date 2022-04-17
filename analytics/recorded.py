import pandas as pd
from matplotlib import pyplot as plt

plt.style.use('seaborn')

data = pd.read_csv('../recordedData.csv')

date = data['date']

averagePerson = data['averagePerson']
averageViolator = data['averageViolator']
averageViolation = data['averageViolation']

plt.plot_date(date, averagePerson, linestyle = 'solid', label='Average Person')
plt.plot_date(date, averageViolator, linestyle = 'solid', label='Average Violator')
plt.plot_date(date, averageViolation, linestyle = 'solid', label='Average Violation')

plt.legend(prop={'size': 12}, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)

plt.title('Recorded Data Visualization')
plt.xlabel('Date')
plt.ylabel('Data')

plt.tight_layout()

plt.show()





