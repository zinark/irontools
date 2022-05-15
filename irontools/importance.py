from textwrap import indent
from sklearn import set_config
from IPython.core.display import display, HTML, Markdown
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
import ipywidgets  as widgets
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def important_feats (df : pd.DataFrame):
  from sklearn.impute import SimpleImputer
  data = SimpleImputer().fit_transform(df.values)
  df = pd.DataFrame(data, columns=df.columns)
  features = df.select_dtypes("number").columns.tolist()
  
  w_target = widgets.Dropdown(options=features, description="target feat:")
  w_nestimators = widgets.IntSlider(value=25, max=1000, min=25, step=25)
  w_alg = widgets.Dropdown(options=['pearson', 'kendall', 'spearman'], description="algorithm:")

  def plot (target, n_estimators):
    w = widgets.IntProgress(description="fitting...")
    X = df.drop(target, axis=1)
    y = df[target]
    display(w)
    model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, max_depth=5)
    model.fit(X, y)

    df_imp = pd.DataFrame({
      "feat" : X.columns.tolist(),
      "importance" : model.feature_importances_
    }).set_index("feat").sort_values(by="importance")
    df_imp.plot.barh()
    w.close()

    

    

  
  widgets.interact(plot, target=w_target, n_estimators=w_nestimators)