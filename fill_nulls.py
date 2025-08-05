def column_transformations(df, columns_list):
    # Keep rows with missing values in 'price', 'bed', or 'bath' while applying conditions

    # Remove rows where the 'price' <= 1000 and greater than the max allowed value (but keep NaNs)
    if "price" in columns_list:
        df = df[
            ((df["price"] > 1000) & (df["price"] <= (285 * 10**6))) | df["price"].isna()
        ]

    # Remove rows where 'bed' > 25 (but keep NaNs)
    if "bed" in columns_list:
        df = df[(df["bed"] <= 25) | df["bed"].isna()]

    # Remove rows where 'bath' > 20 (but keep NaNs)
    if "bath" in columns_list:
        df = df[(df["bath"] <= 20) | df["bath"].isna()]

    # Reset index after filtering
    df = df.reset_index(drop=True)

    return df


def get_unique_city(x):
    unique_vals = x.dropna().unique()
    return unique_vals[0] if len(unique_vals) == 1 else np.nan


def get_mode_zip(x):
    vc = x.value_counts(normalize=True)

    if not vc.empty and vc.iloc[0] > 0.95:
        return vc.index[0]
    return np.nan


def dynamic_bins(group, na_column, compare_column, conf_count_per_bin=20):
    """Create bins based on quantiles for each (zip_code, city, state) group."""

    grp_values = group[compare_column].dropna()
    if len(grp_values) == 0:
        group["range"] = "All values are NaN"
        group["median"] = np.nan

        return group

    quantiles = np.quantile(grp_values, [0, 0.25, 0.5, 0.75, 1])
    quantiles = np.sort(np.unique(quantiles))

    if len(quantiles) == 1:
        group["range"] = f"Constant {quantiles[0]:.2f}"
        if group[na_column].notna().sum() >= conf_count_per_bin:
            group["median"] = group[na_column].median()
        else:
            group["median"] = np.nan

        return group

    # Define the bins
    bin_labels = [
        f"{quantiles[i]:.2f}-{quantiles[i+1]:.2f}" for i in range(len(quantiles) - 1)
    ]

    # Create the bins using pd.cut
    group["range"] = pd.cut(
        group[compare_column],
        bins=quantiles,
        labels=bin_labels,
        include_lowest=True,
        duplicates="drop",
    )

    # Assign the median value to each bin if the bin has more than 10 values, otherwise NaN
    median_values = group.groupby("range", observed=True)[na_column].apply(
        lambda x: x.median() if x.notna().sum() >= conf_count_per_bin else np.nan
    )

    # Map the calculated medians to the 'range' column
    group["median"] = group["range"].map(median_values)

    return group


def fill_missing_values_with_comparison(
    df, na_column, compare_column, conf_count_per_bin=25
):

    grouped_df = df.groupby(["zip_code", "city", "state"], group_keys=False)[
        df.columns.tolist()
    ].apply(
        dynamic_bins,
        na_column=na_column,
        compare_column=compare_column,
        conf_count_per_bin=conf_count_per_bin,
    )

    grouped_df[na_column] = grouped_df[na_column].fillna(grouped_df["median"])
    grouped_df = grouped_df.drop(["median", "range"], axis=1)
    return grouped_df


def pipeline(df, columns_list):
    # state
    if "state" in columns_list:
        df = df.dropna(subset=["state"])

    # city
    if "city" in columns_list:
        df.loc[:, "city"] = df["city"].fillna(
            df.groupby(["state", "zip_code"])["city"].transform(get_unique_city)
        )
        df = df.dropna(subset=["city"])

    # zip_code
    if "zip_code" in columns_list:
        df["zip_code"] = df["zip_code"].fillna(
            df.groupby(["city", "state"])["zip_code"].transform(get_mode_zip)
        )
        df = df.dropna(subset=["zip_code"])

    # drop prev_sold_date, street
    if "street" in columns_list:
        df = df.drop(["street"], axis=1)

    if "prev_sold_date" in columns_list:
        df = df.drop(["prev_sold_date"], axis=1)

    if "status" in columns_list:
        df = df.drop(["status"], axis=1)

    # brokered_by
    if "brokered_by" in columns_list:
        df["brokered_by"] = df["brokered_by"].fillna(999999)

    # bed
    if "bed" in columns_list:
        df = fill_missing_values_with_comparison(df.copy(), "bed", "house_size", 25)

    # bath
    if "bath" in columns_list:
        df = fill_missing_values_with_comparison(df.copy(), "bath", "house_size", 25)

    # price (will be better if use linear reg to estimate the coeffs)
    if "price" in columns_list:
        df["total_area_sqft"] = df["house_size"] + df["acre_lot"] * 43560
        df = fill_missing_values_with_comparison(
            df.copy(), "price", "total_area_sqft", 25
        )

        # drop
        df = df.drop(["total_area_sqft"], axis=1)

    not_required_columns = set(df.columns) - set(columns_list)
    df = df.drop(not_required_columns, axis=1)

    # drop na from all other columns
    df = df.dropna()

    # reset index
    df = df.reset_index(drop=True)
    return df
