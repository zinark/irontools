from sklearn import set_config
from IPython.core.display import display, HTML
from sklearn.utils import estimator_html_repr
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline, FeatureUnion
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
set_config(display='diagram')
plt.style.use('fivethirtyeight')

def init_jupyter():
  print ("add following to your jupyter to install")
  print ("!pip install scikit-learn==1.1.0rc1 lightgbm catboost xgboost shap numpy pandas")
  print ("scikit-learn", sklearn.__version__)
  print ("pandas", pd.__version__)
  print ("numpy", np.__version__)

def describe(df : pd.DataFrame):
  print (len(df))
  print (df.shape)
  print ("describe")
