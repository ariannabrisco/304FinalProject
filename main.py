# IMPORTS
import csv
import matplotlib.pyplot as plt
import numpy as np
# my PCA function from unit 1 homework
from helper import createPCA

# FUNCTION DEFINITIONS
# save data from CSV
def readWineCSV(filename):
    data = []
    labels = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:
                labels.append(int(row[0]))                  # save for plotting
                features = [float(x) for x in row[1:]]      # column 0 is class of wine
                data.append(features)
    return data, labels


# FUNCTION CALLS
wineData, wineLabels = readWineCSV("wineData.csv")
winePCs = createPCA(wineData, 2)

# printing to check head
print("Principal Components (first 5 samples):")
print(np.array(winePCs[:5]))


# PLOT DATA
# create arrays of principal components and labels (class of wine)
winePCS = np.array(winePCs)
wineLabels = np.array(wineLabels)

# Workaround -- plot one class at a time with different colors for each class
for class_id in np.unique(wineLabels):
    idx = wineLabels == class_id            # gross notation because not using pandas
    plt.scatter(winePCs[idx, 0], winePCs[idx, 1], alpha=0.5, label=f"Class {class_id}")

# Design plot
plt.title("PCA of Wine Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()