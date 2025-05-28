import pandas as pd

# Baca file Excel (.xlsx)
df = pd.read_excel("Dataset_resep.xlsx")

# Simpan ke file CSV dengan delimiter koma ,
df.to_csv("dataset_resep.csv", index=False, sep=",", encoding="utf-8-sig")
