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

def corr (df : pd.DataFrame):
  features = df.select_dtypes("number").columns.tolist()
  w = widgets.Dropdown(options=features, description="target feat:")
  w_thresh = widgets.FloatSlider(value=0.3, max=1, min=0)
  w_alg = widgets.Dropdown(options=['pearson', 'kendall', 'spearman'], description="algorithm:")

  def plot (col, method, thresh):
    corr = pd.DataFrame(df.corr(method=method)[col].drop(col, axis=0).sort_values())
    corr = corr[corr.abs() > thresh].dropna()
    display(corr)
    corr.plot.barh(figsize=(12,8))
  
  widgets.interact(plot, col=w, thresh=w_thresh, method=w_alg)
  





def corr_samples (df: pd.DataFrame):
  features = df.select_dtypes("number").columns.tolist()
  w1 = widgets.Dropdown(options=features, description="feat 1st:")
  w2 = widgets.Dropdown(options=features, description="feat 2st:")
  w_sort = widgets.Checkbox(value=True, description="Sort 2st:", indent=False)

  def plot (col1, col2,sorted):
    
    
    data = df[[col1, col2]]
    if col1 == col2:
      data =df[[col2]]

    if sorted:
      data = data.sort_values(by=col2)
    
    data = data.reset_index().drop("index", axis=1)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12,8))
    data[[col1]].plot(ax=axes[0], linewidth=1, alpha=0.5)
    data[[col2]].plot(ax=axes[1], linewidth=1, alpha=0.5)
  
  widgets.interact(plot,col1=w1, col2=w2, sorted=w_sort)
  








def preprocess_categorics (df):
  return pd.get_dummies(df)
















def describe_feats(df : pd.DataFrame):
  display(Markdown(f"# Data Shape {df.shape}"))

  num_cols = df.select_dtypes("number").columns.tolist()
  cat_cols = df.select_dtypes("object").columns.tolist()
  feature_types = ["numeric", "categoric"]

  features = df.columns
  w1 = widgets.Dropdown(options=features, description="feats")

  def plot (col):

    # NUMERIC PLOTS
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
    
    # CATEGORIC PLOTS
    if col in df.select_dtypes("object").columns:
      fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
      df[[col]].value_counts().plot.barh(ax=axes[0])
      (df[[col]].value_counts()/len(df)).plot.barh(ax=axes[1])
      # axes[0].vline(0.5)
      plt.show()
      
      df_summary = df[[col]].value_counts().reset_index()
      df_summary.columns = ["feat", "count"]
      df_summary["percentage"] = df_summary["count"] / len(df)
      df_summary["percentage"] = df_summary["percentage"].round(3)
      
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

  
  
