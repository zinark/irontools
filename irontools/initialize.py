def init():
  print ("<<<< Required packages >>>")
  print ("!pip install scikit-learn==1.1.0rc1 lightgbm catboost xgboost shap numpy pandas ipywidgets matplotlib")
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
  import ipywidgets as widgets
  set_config(display='diagram')
  plt.style.use('fivethirtyeight')
  print ("scikit-learn", sklearn.__version__)
  print ("pandas", pd.__version__)
  print ("numpy", np.__version__)
  import matplotlib.pyplot as plt
  plt.rcParams['figure.figsize'] = (20, 4)

  