import pandas as pd
df = pd.read_csv("data/final_sarc_cohort.csv")
print("🏆 FULL COHORT STATS")
print(df.describe())
print("\nTop 10 patients:")
print(df.nlargest(10, 'mutations')[['patient', 'mutations', 'candidates', 'fanout']])
print("\nHigh TMB:", df.loc[df['mutations'].idxmax()])
df.to_excel("data/sarc_cohort.xlsx", index=False)
print("\nExcel saved: data/sarc_cohort.xlsx")