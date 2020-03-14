

from matplotlib import pyplot as plt
from numpy import cov



def display_scatter_plot(feature1, feature2):
    plt.scatter(feature1, feature2)
    plt.show()

def calculate_correlation(feature1, feature2):
    return cov(feature1, feature2)