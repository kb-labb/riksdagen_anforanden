This folder contains the adjusted metadata output from KBLab. The data contains ids for the debates and speeches, as well as the start and end times for each speech. 

* `adjusted_metadata.csv.gz`: The approximately 117000 speeches for which we have high confidence in the metadata adjustments.
* `missing_audio_text.csv.gz`: Approximately 10000 speeches for which there is downloadable media via the API, but either the audio is corrupt, or too short to actually contain the debate/speeches. 
* `low_quality_metadata.csv.gz`: Approximately 4000 speeches for which we have low confidence. Mostly speeches where the debate audio file begins or ends in the middle of a speech, or where the speech is very short. But also some speeches with duplicate official transcripts, as well as speeches from audio/media files that contain encoding errors causing the speech to be partially cut off or skipped.

Guide for metadata fields:

* `dokid`: Document id for the debate. This is the same for all speeches in a debate.
* `anforande_nummer`: Speech number within the debate, or debate within sessions in a particular day. Should create a unique primary key combination together with `dokid`, but sometimes there are duplicates. 
* `start`: The Riksdag's start time of speech in seconds.
* `duration`: The Riksdag's duration of speech in seconds.
* `end`: The Riksdag's end time of speech in seconds.
* `start_adjusted`: KBLab's adjusted start time of speech in seconds.
* `end_adjusted`: KBLab's adjusted end time of speech in seconds.
* `speaker`: The speaker's name retrieved via the `intressent_id`. 
* `party`: The speaker's party retrieved via the `intressent_id`.
* `gender`: The speakers gender retrieved via the `intressent_id`.
* `text`: The speaker's name as listed in  text formar (sometimes wrong and mismatched against `intressent_id`).) 
* `intressent_id`: Unique id for the speaker within the Riksdag's database.
* `is_prediction`: We reverted back to the Riksdag's metadata for a small subset of modern speeches (500 out of 117k). A `True` value of `is_prediction` indicates that the speech was predicted by KBLab. 
* `debatedate`: The date of the debate.