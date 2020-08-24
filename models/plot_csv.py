import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


if len(sys.argv) <= 1:
    print('need filename as argument')

filename = sys.argv[1]

with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    steps = []
    values = []
    value_id = None
    for row in csv_reader:
        if value_id is None:
            value_id = row[1]
            continue
        steps.append(int(row[0]))
        values.append(float(row[1]))

    plt.scatter(np.array(steps), np.array(values))

    filename = filename.split('.')[0]
    plt.title('Scatter plot of ' + filename)
    plt.xlabel('# training steps')
    plt.ylabel(value_id)

    # set number of labels on x and y axis
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=10)

    x1,x2,y1,y2 = plt.axis()
    plt.axis((None,None,None,None))
    
    plt.savefig(filename + '.png')
