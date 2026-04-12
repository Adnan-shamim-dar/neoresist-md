import pandas as pd

df = pd.read_parquet("C:/Users/rambe/Documents/neovax/data/final/enriched_candidates.parquet")
print(df[['purity','purity_source_used','purity_method_used','purity_resolution_reason']].drop_duplicates().head(20))