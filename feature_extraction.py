import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from dataset_builders import build_test_train_set

#
# Uses XGBoost to determine feature importance on a subject's data from the WESAD dataset.
# Returns a dataframe containing all signals correlated to their feature importance.
#
def use_xgb_regressor(_df: pd.DataFrame, show_chart: bool = True):
  cleaned_df = _df.dropna(how='all', subset=[col for col in _df.columns if col != 'label'])
  
  (train_df, test_df) = build_test_train_set(cleaned_df)

  Xtr = train_df.drop(columns=['label'], inplace=False)
  Xts = test_df.drop(columns=['label'], inplace=False)

  ytr = train_df['label'].values.ravel()
  yts = test_df['label'].values.ravel()

  reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=None)
  reg.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xts, yts)], verbose=False)

  if reg.get_booster() is None:
    raise RuntimeError("The XGBoost model has no booster. Training might have failed.")


  if show_chart == True:
    xgb.plot_importance(reg.get_booster(), height=0.7, alpha=0.7)
    plt.show()

  fscore_importance = reg.get_booster().get_score(importance_type='weight')

  feature_importance_df = pd.DataFrame({
    'Feature': list(fscore_importance.keys()),
    'Importance': list(fscore_importance.values())
  })
  feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

  return feature_importance_df

#
# Using `use_xgb_regressor`, evaluates all signals and their feature importance across all subjects and averages them together.
# Returns a Dataframe where all signals are correlated with their feature importance across all subject data.
#
def get_feature_importance(data_dict: dict):
  _total_df = pd.DataFrame()
  for entry in data_dict.keys():
    _df = use_xgb_regressor(data_dict[entry], show_chart=False)
    if _total_df.empty:
      _total_df = _df
    else:
      _total_df['Importance'] = _total_df['Importance'] + _df['Importance']

  _total_df['Importance'] = _total_df['Importance'] / len(data_dict.keys())
  return _total_df