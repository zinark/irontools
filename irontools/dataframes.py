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

def describe_feats(df : pd.DataFrame):
  display(Markdown(f"# Data Shape {df.shape}"))

  num_cols = df.select_dtypes("number").columns.tolist()
  cat_cols = df.select_dtypes("object").columns.tolist()
  feature_types = ["numeric", "categoric"]

  features = df.columns
  w1 = widgets.Dropdown(options=features, description="feats")
  
  def plot (col):

    if col in df.select_dtypes("number").columns:
      std = df[col].std().round(3)
      mean = df[col].mean().round(3)
      display (Markdown(f"*{col}* : mean=**{mean}**±{std}"))
      fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12,8))
      df[[col]].boxplot(ax=axes[0], vert=False)
      df[[col]].plot.hist(ax=axes[1], bins=50)
      df[[col]].plot.kde(ax=axes[2])
      
      # df[[col]].boxplot(ax=axes[0], vert=False)
      # df[[col]].plot.hist(ax=axes[1], bins=50)
      # df[[col]].plot.kde(ax=axes[2])
      # plt.tight_layout()
      return

    if col in df.select_dtypes("object").columns:
      fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
      df[[col]].value_counts().plot.barh()
      plt.show()
      
      df_summary = df[[col]].value_counts().reset_index()
      df_summary.columns = ["feat", "count"]
      display(df_summary)
      return
    
    print ("bilinmeyen tip")

  widgets.interact(plot, col=w1)

  # @widgets.interact(which_features=which_cols)
  # def render_parent (which_features):
  #   print (which_features)
  #   cols = num_cols

  # @widgets.interact(cols=cols)
  # def render (cols):
  #   print (cols)

  # @widgets.interact(
  #   which_feature=which_cols,
  #   num_col=num_cols, 
  #   sort=True, 
  #   target_col=num_cols, 
  #   norm=True
  #   )
  # def render (which_feature, num_col, sort, target_col, norm):
  #     df_data = df[[num_col, target_col]]
  #     if sort:
  #         df_data = df_data.sort_values(by=num_col)
      
  #     if norm:
  #         from sklearn.preprocessing import StandardScaler
  #         data = StandardScaler().fit_transform(df_data)
  #         df_data = pd.DataFrame(data, columns=df_data.columns)
  #     df_data.plot(alpha=0.2)
  #     plt.show()

  
  
