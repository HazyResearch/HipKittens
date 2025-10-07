import pandas as pd

# Load the CSV
try:
    df = pd.read_csv("out/hit_rate_counter_collection.csv")
except FileNotFoundError:
     df = pd.read_csv("hit_rate_counter_collection.csv")

# Keep only LDS-related counters
df = df[df["Counter_Name"].isin(["TCP_TOTAL_CACHE_ACCESSES_sum"])]

# Pivot so each dispatch becomes one row with both counters
pivot = df.pivot_table(index=["Dispatch_Id", "Kernel_Name"],
                       columns="Counter_Name",
                       values="Counter_Value",
                       aggfunc="first").reset_index()

# Replace NaNs or div-by-zero
pivot = pivot.fillna(0)

# Show only interesting kernels
# interesting = pivot[pivot["SQ_INSTS_LDS"] > 0]
interesting = pivot[pivot['Kernel_Name'].str.contains('micro_tk')]

# Print summary
pd.set_option('display.max_colwidth', 100)
print(interesting[["Dispatch_Id", "Kernel_Name", "TCP_TOTAL_CACHE_ACCESSES_sum"]])