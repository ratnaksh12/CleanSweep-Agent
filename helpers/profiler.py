def profile_dataset(df):
    profile = []

    for col in df.columns:
        dtype = df[col].dtype
        nulls = df[col].isnull().sum()
        unique = df[col].nunique()
        profile.append({
            "Column": col,
            "Type": str(dtype),
            "Nulls": nulls,
            "Unique Values": unique
        })

    return profile
