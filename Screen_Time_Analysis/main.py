import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("screentime_analysis.csv")

data['Date'] = pd.to_datetime(data['Date'])

plt.figure(figsize=(12,6))
sns.lineplot(x='Date', y='Usage (minutes)', hue='App', data=data, marker='o')
plt.title('Screen time analysis ')
plt.ylabel('Usage (minutes)')
plt.xlabel('Date')
plt.xticks(rotation=22.5)
plt.tight_layout()
plt.show()
