"""
Example of what a mixed effects model script might look like for analysing results.
"""

import pandas as pd
from statsmodels.formula.api import mixedlm

df = pd.read_csv("evaluation_results.csv")
model = mixedlm("score ~ temperature + agent", df, groups=df["seed"])
result = model.fit()
print(result.summary())
