import pandas as pd
from glob import glob

dataset = 'cancer'
out_df_list = []
for out_csv in glob(f"reports/{dataset}/*.csv"):
    out_df_list.append(pd.read_csv(out_csv))
out_df = pd.concat(out_df_list, ignore_index=True)
out_df.to_csv(f"report_{dataset}.csv", index=False)