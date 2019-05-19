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

# Ensemble learning with Averaging
sub1.label = (sub1.label + sub2.label + sub3.label)/3

# Create submission file
sub1.to_csv('ensemble_FINAL1.csv', index=False)