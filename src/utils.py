def print_overlapping_segments(df, row_nr, method="fuzzy", column="anftext_inference"):
    """
    Prints segments that overlap between actual transcription ("anftext_normalized")
    and inferred transcription ("anftext_inference"). Overlapping segments are
    colored in green. Non-overlapping segments are colored in red.

    When column="anftext_inference", we print "anftext_inference".
    The parts of the the inferred transcription ("anftext_inference") that are also
    found/matched in the actual transcription ("anftext_normalized") are colored in green.

    Args:
        df (pd.DataFrame): DataFrame with columns "anftext_normalized" and "anftext_inference".
        row_nr (int): Row number of the DataFrame to print.
        method (str): Method to use for matching. Either "ngram" or "fuzzy".
        column (str): Column to print. Either "anftext_inference" or "anftext_normalized".
    """

    if "fuzzy" in method and column == "anftext_inference":
        start = int(df["start_fuzzy_i"][row_nr])
        end = int(df["end_fuzzy_i"][row_nr])
    elif "fuzzy" in method and column == "anftext_normalized":
        start = int(df["start_fuzzy_a"][row_nr])
        end = int(df["end_fuzzy_a"][row_nr])
    elif "ngram" in method and column == "anftext_inference":
        start = int(df["start_ngram_i"][row_nr])
        end = int(df["end_ngram_i"][row_nr])
    elif "ngram" in method and column == "anftext_normalized":
        start = int(df["start_ngram_a"][row_nr])
        end = int(df["end_ngram_a"][row_nr])

    split_text = df[column][row_nr].split()
    color_text = (
        ["\033[91m"]
        + split_text[0:start]  # red
        + ["\033[0m"]
        + ["\033[92m"]
        + split_text[start:end]  # green
        + ["\033[0m"]
        + ["\033[91m"]
        + split_text[end:]  # red
        + ["\033[0m"]
    )
    overlapping_text = " ".join(color_text)

    print(overlapping_text)
