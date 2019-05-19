# Importing libraries
import os
import numpy as np
import pandas as pd
import scipy.special
import statistics
from statistics import mode
from scipy.stats import mode

# Using sigmoid function as activation function
sigmoid = lambda x:scipy.special.expit(x)

os.listdir('C:/Users/Shruti/Desktop')

# Submission files of previous approaches 
sub1 = pd.read_csv('C:/Users/Shruti/Desktop/Submission_ONE.csv')
sub2 = pd.read_csv('C:/Users/Shruti/Desktop/SubmissionTwo.csv')
sub3 = pd.read_csv('C:/Users/Shruti/Desktop/ensembleML.csv')

sub1.head()

sub2.head()

sub3.head()

# Assuming weights for the above submission files
w1 = 0.4
w2 = 0.4
w3 = 0.2

# Ensemble learning with Weighted Averaging
sub1.label = (w1 * sub1.label) + (w2 * sub2.label) + (w3 * sub3.label)

# Create submission final
sub1.to_csv('ensemble_FINAL2.csv', index=False)