import multiprocessing as mp

import numpy as np
from rapidfuzz import fuzz
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import TreebankWordTokenizer
from nltk import ngrams
from tqdm import tqdm


def calculate_bleu(text1, text2):
    """
    Calculate BLEU score between two texts.
    """

    if text1 is None or text2 is None:
        return None
    else:
        chencherry = SmoothingFunction()
        return sentence_bleu(
            references=[text1.split()],
            hypothesis=text2.split(),
            smoothing_function=chencherry.method4,
        )


def get_ngrams_array(text, n):
    """
    Returns an array of ngrams for a given text

    args:
        text: str
        n: int
    """
    n_grams = ngrams(text.split(), n)
    n_grams = np.array(list(n_grams))
    return n_grams


def get_ngram_index_match(anftext_inference_ngrams, ngram):
    """
    Get ngrams for anftext_inference and check if argument ngram is in text_inference.
    Returns boolean array with True value(s) where the given ngram match is found

    Args:
        anftext_inference_ngrams (np.array): Ngrams of text transcription of speech
            audio file by wav2vec2. Via function get_ngrams_array().
        ngram (list | tuple | np.array): Ngram to check if it is in anftext_inference

    Returns:
        np.array: Boolean array with True value(s) where the given ngram match is found
    """

    ngram_bool_matrix = anftext_inference_ngrams == tuple(ngram)

    if ngram_bool_matrix is bool:
        return np.array(ngram_bool_matrix)
    else:
        return ngram_bool_matrix.all(axis=1)


def get_weighted_ngram_score(anftext_normalized, anftext_inference, n):
    """
    Get weighted ngram scores for anftext_inference and anftext_normalized using
    ngrams of several different sizes (from 1 to n).

    Args:
        anftext_normalized (str): Official normalized text transcription of speech audio file.
        anftext_inference (str): Text transcription of speech audio file by wav2vec2.
        n (int): Maximum ngram size. Will use 1 to n ngram size.

    Returns:
        np.array: Array with different ngram size occurences weighted together.
    """

    ngrams_bool_list = []
    for i in range(1, n):
        anftext_normalized_ngrams = get_ngrams_array(anftext_normalized, n=i)
        anftext_inference_ngrams = get_ngrams_array(anftext_inference, i)
        array_list = []
        for ngram in anftext_normalized_ngrams:
            array_list.append(get_ngram_index_match(anftext_inference_ngrams, ngram))

        ngram_matches = np.vstack(array_list)
        # Which indices in anftext_inference_ngrams that matches anftext_normalized_ngrams
        ngram_matches = ngram_matches.any(axis=0)
        # Add some zeroes at the end because ngrams of different size are not the same length
        ngram_matches = np.concatenate([ngram_matches, np.zeros(i - 1, dtype=bool)])
        ngram_matches = np.convolve(
            ngram_matches, np.ones(i + 2) * np.sqrt(np.log(i + 1)), mode="same"
        )  # Longer convolutions and higher weights for longer ngrams
        ngrams_bool_list.append(ngram_matches)

    ngram_matches = np.vstack(ngrams_bool_list)
    ngram_matches = ngram_matches.sum(axis=0)
    ngram_matches = ngram_matches / 3

    return ngram_matches


# Source: https://stackoverflow.com/q/4494404
def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    (idx,) = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]  # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx


def contiguous_ngram_match(
    anftext_normalized, anftext_inference, n=6, threshold=1, min_continuous_match=8, max_gap=30
):
    """
    Get (fuzzy-ish) contiguous matching indices for anftext_inference and anftext_normalized
    using ngrams of several different sizes (from 1 to n) to construct weighted scores.

    Args:
        anftext_normalized (str): Official normalized text transcription of speech audio file.
        anftext_inference (str): Text transcription of speech audio file by wav2vec2.
        n (int): Maximum ngram size. Will use 1 to n ngram size.
        threshold (float): Weighted ngram score threshold for being considered a match.
        min_continous_match (int): Minimum continuous word matches for the region to
            be considered contiguous.
        max_gap (int): Maximum gap (in words) between contiguous region and the next/previous
            region for it to be seen as the start/end index of a larger joined together contiguous
            region.

    Returns:
        tuple: Start and end indices of contiguous fuzzy match in anftext_inference
            (or whatever text is input as second arg).
    """

    if anftext_normalized is None or anftext_inference is None or len(anftext_inference.split()) == 1:
        return None, None

    ngram_match_scores = get_weighted_ngram_score(anftext_normalized, anftext_inference, n=n)
    # Contiguous region indices satisfying the condition ngram_match_scores > threshold
    ngram_match_indices = contiguous_regions(ngram_match_scores > threshold)

    start_index = None
    end_index = None

    for i in range(0, len(ngram_match_indices)):
        if ngram_match_indices[i][1] - ngram_match_indices[i][0] > min_continuous_match:
            try:
                next_region_gap = ngram_match_indices[i + 1][0] - ngram_match_indices[i][1]

                if next_region_gap > max_gap:
                    # If gap is larger than max_gap, then this is not the start of a contiguous region
                    continue
            except IndexError:
                pass

            start_index = ngram_match_indices[i][0]
            break

    for i in reversed(range(0, len(ngram_match_indices))):
        if ngram_match_indices[i][1] - ngram_match_indices[i][0] > min_continuous_match:

            try:
                previous_region_gap = ngram_match_indices[i][0] - ngram_match_indices[i - 1][1]

                if previous_region_gap > max_gap:
                    continue
            except IndexError:
                pass

            end_index = ngram_match_indices[i][1]
            break

    if start_index is None or end_index is None:
        return None, None
    else:
        return start_index, end_index


def contiguous_ngram_match_star(args):
    """
    Wrapper for multiprocessing.
    Unpacks arguments and calls contiguous_ngram_match().
    """
    return contiguous_ngram_match(*args)


def get_fuzzy_match_word_indices(anftext_inference, alignment):
    """
    Rapidfuzz and fuzzysearch return character indices for fuzzy matches.
    This function converts them to word indices.
    """

    word_spans = TreebankWordTokenizer().span_tokenize(anftext_inference)
    word_spans = np.array(list(word_spans))

    if alignment.src_start == 0:
        start_index = 0
    else:
        start_index = np.where(word_spans <= alignment.src_start)[0][-1]

    if alignment.src_end == len(anftext_inference):
        end_index = len(word_spans)
    else:
        end_index = np.where(word_spans >= alignment.src_end)[0][0]

    return start_index, end_index


def contiguous_fuzzy_match(anftext_normalized, anftext_inference, threshold=55):
    """
    Fuzzy contiguous index match for anftext_inference and anftext_normalized using
    fuzz.partial_ratio_alignment().

    Args:
        anftext_normalized (str): Official normalized text transcription of speech audio file.
        anftext_inference (str): Text transcription of speech audio file by wav2vec2.
        threshold (int): Fuzzy match score threshold for being considered a match.
            0 to 100, 100 being exact match.

    Returns:
        tuple: Start and end indices of contiguous fuzzy match in anftext_inference,
            along with fuzzy match score of the matching segment.
    """

    if anftext_normalized is None or anftext_inference is None or len(anftext_inference.split()) <= 1:
        return None, None, None

    align = fuzz.partial_ratio_alignment(anftext_inference, anftext_normalized)

    align_check = fuzz.partial_ratio_alignment(anftext_inference[align.src_start : align.src_end], anftext_normalized)

    # Sanity check to make sure the suggested alignment is correct
    if align_check.score < threshold and align.score < threshold:
        return None, None, None

    start_index, end_index = get_fuzzy_match_word_indices(anftext_inference, align)

    return start_index, end_index, align_check.score


def contiguous_fuzzy_match_star(args):
    """
    Wrapper function for multiprocessing.
    Unpacks arguments and calls contiguous_fuzzy_match().
    """
    return contiguous_fuzzy_match(*args)


def contiguous_ngram_indices(
    df,
    column_in,
    column_out,
    n=6,
    threshold=1.3,
    min_continuous_match=8,
    max_gap=30,
    processes=None,
):
    """
    Find and return the indices of the contiguous text in column_out that
    matches text in column_in.

    Args:
        df (pd.DataFrame): DataFrame containing column_in and column_out.
        column_in (str): Column name of text to match against.
        column_out (str): Column name of text we get matching indices for.
            n (int): N-gram sizes 1 to n.
        threshold (float): Threshold score for contiguous n-gram match to be considered a match.
        min_continous_match (int): Minimum continuous word matches for the region to
            be considered contiguous.
        max_gap (int): Maximum gap (in words) between contiguous region and the next/previous
            region for it to be seen as the start/end index of a larger joined together contiguous
            region.
        processes (int | NoneType): Number of processes to use for multiprocessing.
            If None, use all available processes.
    """

    df

    with mp.Pool(processes) as pool:
        args = [
            (text1, text2, n, threshold, min_continuous_match, max_gap)
            for text1, text2 in zip(df[column_in], df[column_out])
        ]
        contiguous_ngram_list = list(
            tqdm(
                pool.imap(
                    contiguous_ngram_match_star,
                    args,
                    chunksize=1,
                ),
                total=len(df),
            )
        )

    return contiguous_ngram_list


def contiguous_fuzzy_indices(
    df,
    column_in,
    column_out,
    threshold=55,
    processes=None,
):
    """
    Find and return the indices of the contiguous text in column_out that
    matches text in column_in.

    Args:
        df (pd.DataFrame): DataFrame containing column_in and column_out.
        column_in (str): Column name of text to match against.
        column_out (str): Column name of text we get matching indices for.
            n (int): N-gram sizes 1 to n.
        threshold (int): Threshold score for the fuzzy match to return indices.
        processes (int | NoneType): Number of processes to use for multiprocessing.
            If None, use all available processes.
    """

    with mp.Pool(processes) as pool:
        args = [(text1, text2, threshold) for text1, text2 in zip(df[column_in], df[column_out])]
        contiguous_fuzzy_list = list(
            tqdm(
                pool.imap(
                    contiguous_fuzzy_match_star,
                    args,
                    chunksize=1,
                ),
                total=len(df),
            )
        )

    return contiguous_fuzzy_list
