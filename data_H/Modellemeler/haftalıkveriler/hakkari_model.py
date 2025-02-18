import os
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from scipy.stats import zscore

df = pd.read_csv("hakkari_end.csv")
print (df.info())
print (df.head())
print(df)


