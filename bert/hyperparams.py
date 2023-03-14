# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
# %%
import pandas as pd

df = pd.read_json("hyperparams.json", typ='series').to_frame()
df.index.name = 'Parameter'
df.columns = ['Value']
with open("hyperparams.txt", 'w') as f:
    print(df.to_latex(), file=f)
