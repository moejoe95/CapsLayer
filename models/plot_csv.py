import sys
import csv
import numpy as np
import pandas as pd
import os.path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

root = 'models/results/fashion_mnist_ResCaps'

path_names = []
i = 0
while os.path.isdir(root + '_' + str(i)):
    path_names.append(root + '_' + str(i))
    i += 1

for path_name in path_names:
    print('plotting: ', path_name)
    try:
        train = pd.read_csv(path_name + '/' + 'train_acc.csv')
        test = pd.read_csv(path_name + '/' + 'val_acc.csv')
    except:
        print('failed to plot')
        continue

    plt.scatter(train['step'], train['train_acc.csv'], label='train')
    plt.scatter(test['step'], test['val_acc.csv'], label='validation')

    plt.title('Scatter plot of ' + path_name)
    plt.xlabel('# training steps')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")

    # set number of labels on x and y axis
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=10)

    x1,x2,y1,y2 = plt.axis()
    plt.axis((None,None,0.0,1.0))
    
    filename = path_name.split('/')[-1]
    plt.savefig('models/results/plots/' + filename + ".png")
    plt.close()

print('finished ...')
