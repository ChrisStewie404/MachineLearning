import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

data = pd.read_csv('./results/mlp_shape.csv')
data = data.to_numpy()[0]
for i,num in enumerate(data):
    results = re.findall(r'\d+.\d+',num)
    data[i] = float(results[0])
plt.figure()
plt.plot(range(2,17),data,'-o')
plt.xticks(range(2,17))
plt.xlabel('Number of neurons in the first hidden layer')
plt.ylabel('Testing Accuracy (%)')
plt.title('MLP Classifier Performance vs. Hidden Layer Size')
plt.grid()
plt.savefig('./results/mlp_shape.png')
plt.show()