

import shap
import xgboost
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
import pandas as pd
import matplotlib as mpl
from matplotlib import cm
import numpy as np

plt.style.use('tableau-colorblind10')
plt.style.use('fivethirtyeight')

data = pd.DataFrame(shap.datasets.nhanesi()[0])
data = data.iloc[:,1:]
data['y'] = shap.datasets.nhanesi()[1]
data.head()

X,y = shap.datasets.nhanesi()
X_display,y_display = shap.datasets.nhanesi(display=True) # human readable feature values

xgb_full = xgboost.DMatrix(X.iloc[:, 1:], label=y)

# create a train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
xgb_train = xgboost.DMatrix(X_train, label=y_train)
xgb_test = xgboost.DMatrix(X_test, label=y_test)

params = {
    "eta": 0.002,
    "max_depth": 3,
    "objective": "survival:cox",
    "subsample": 0.5
}
model_train = xgboost.train(params, xgb_train, 10000, evals = [(xgb_test, "test")], verbose_eval=1000)

# use validation set to choose # of trees
params = {
    "eta": 0.002,
    "max_depth": 3,
    "objective": "survival:cox",
    "subsample": 0.5
}
model = xgboost.train(params, xgb_full, 5000, evals = [(xgb_full, "test")], verbose_eval=1000)

shap_values = shap.TreeExplainer(model).shap_values(X.iloc[:, 1:])
plt.figure(dpi=1200)
shap.dependence_plot("BMI", shap_values, X.iloc[:, 1:], display_features=X_display, show=False, cmap = cm.winter)
pl.xlim(15,50)
pl.show()

shap.summary_plot(shap_values, X.iloc[:, 1:], cmap = cm.winter, plot_type = 'bar', max_display=10)

shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(X.iloc[:3000,1:])

shap.dependence_plot(
    ("Age", "Sex"),
    shap_interaction_values, 
    X.iloc[: 3000,1:],
    display_features=X_display
)


shap.dependence_plot("BMI", shap_values, X.iloc[:, 1:], cmap = cm.winter)

explainer = shap.TreeExplainer(model)
shap.force_plot(explainer.expected_value, shap_values[5,:], X.iloc[5,1:], show=True, 
                matplotlib=True, text_rotation=90, link='identity', plot_cmap="PkYg")

explainer = shap.Explainer(model, X.iloc[:, 1:])
shap_values = explainer(X.iloc[:, 1:])

shap.plots.bar(shap_values[0], max_display = 5)

exp = explainer.explain_instance(X.iloc[1,1:], model, num_features=5)

shap_values[0] = explainer(X.iloc[:, 1:])
sex = ["Women" if shap_values[i,"Sex"].data == 0 else "Men" for i in range(shap_values.shape[0])]
shap.plots.bar(shap_values(sex).abs.mean(0))
shap_values.index[sex]

shap.summary_plot(np.random.randn(20, 5), np.random.randn(20, 5), plot_type="bar", show=False)