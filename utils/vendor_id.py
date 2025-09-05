import numpy as np
import pandas as pd

def assign_vendor_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Assign vendor ids to each feature."""
    print("=" * 80)
    print("📌 [Vendor ID Assignment] Starting vendor assignment process")

    #step-1 create the three unique vendor ids
    vendor_ids = ["VENDOR_1", "VENDOR_2", "VENDOR_3"]
    print(f"Created vendor IDs: {vendor_ids}")

    # Step 2: Assign vendor IDs randomly to each row
    df["vendor_id"] = np.random.choice(vendor_ids, size=len(df))
    print(f"Assigned vendor IDs to {len(df)} rows")

    # Step 3: Show distribution of rows per vendor
    distribution = df["vendor_id"].value_counts()
    print("📊 Vendor distribution in dataset:")
    print(distribution)

    print("✅ [Vendor ID Assignment] Completed vendor assignment")
    print("=" * 60)

    return df