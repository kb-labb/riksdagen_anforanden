import pandas as pd

pd.set_option("display.max_colwidth", 95)

df = pd.read_parquet("data/df_inference_bleu_eval.parquet")
df_meta = pd.read_parquet("data/df_final_metadata.parquet")

df = df_meta.merge(
    df[["dokid", "anforande_nummer", "contiguous_fuzzy_match", "anftext_normalized", "bleu_score"]],
    on=["dokid", "anforande_nummer"],
    how="left",
)

df["fuzzy_score_speech"] = df["contiguous_fuzzy_match"].apply(lambda x: x[2])
df["start_fuzzy_speech"] = df["contiguous_fuzzy_match"].apply(lambda x: int(x[0]))
df["end_fuzzy_speech"] = df["contiguous_fuzzy_match"].apply(lambda x: int(x[1]))

# Count the number of words in anftext_inference
df["nr_words_inference"] = df["anftext_normalized"].str.split().str.len()

# Group by dokid, set first obs in each group as True
df["first_speech"] = df.groupby("dokid")["anforande_nummer"].transform("first") == df["anforande_nummer"]
# Set last obs in each group as True
df["last_speech"] = df.groupby("dokid")["anforande_nummer"].transform("last") == df["anforande_nummer"]

df["end_fuzzy_diff"] = df["end_fuzzy_speech"] - df["nr_words_inference"]

df = df[
    ~(
        (
            (
                ((df["end_fuzzy_diff"] <= -20) & (df["debatedate"] <= "2012-01-01") & df["last_speech"])
                | ((df["end_fuzzy_diff"] <= -30) & (df["debatedate"] > "2012-01-01") & df["last_speech"])
                | ((df["end_fuzzy_diff"] <= -25) & (df["debatedate"] <= "2012-01-01")) & (~df["last_speech"])
                | ((df["end_fuzzy_diff"] <= -35) & (df["debatedate"] > "2012-01-01")) & (~df["last_speech"])
            )
        )
        | (
            ((df["start_fuzzy_speech"] > 30) & (df["debatedate"] > "2012-01-01"))
            | ((df["start_fuzzy_speech"] > 20) & (df["debatedate"] <= "2012-01-01") & (~df["first_speech"]))
            | ((df["start_fuzzy_speech"] > 4) & (df["debatedate"] < "2006-01-01") & (df["first_speech"]))
            | (
                (df["start_fuzzy_speech"] >= 15)
                & (df["debatedate"] > "2006-01-01")
                & (df["debatedate"] <= "2012-01-01")
                & (df["first_speech"])
            )
        )
        | (df["bleu_score"] < 0.2)
    )
].reset_index(drop=True)

df.to_parquet("data/df_final_riksvox.parquet", index=False)
