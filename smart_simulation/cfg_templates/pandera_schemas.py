import pandera as pa

weight_series = pa.SeriesSchema(pa.Float64, index=pa.Index(pa.DateTime), name="weight")
consumption_series = pa.SeriesSchema(
    pa.Float64, index=pa.Index(pa.DateTime), name="consumption"
)
servings_series = pa.SeriesSchema(pa.Int, index=pa.Index(pa.DateTime), name="servings")
