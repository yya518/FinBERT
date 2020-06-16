# FinBERT

`FinBERT` is a BERT model trained on financial communication text. It results in state-of-the-art performance on financial sentiment classification task, which is a core financial NLP task.

`FinBERT` is trained on the following three finanical communication corpus. The total corpora size is 4.9B tokens.

* Corporate Reports 10-K & 10-Q: 2.5B tokens
* Earnings Call Transcripts: 1.3B tokens
* Analyst Reports: 1.1B tokens

## Download FinBERT

We provide four versions of pre-trained weights. 
- FinBERT-FinVocab-Uncased (**Recommended**)
- FinBERT-FinVocab-Cased
- FinBERT-BaseVocab-Uncased
- FinBERT-BaseVocab-Cased

`FinVocab` is a new WordPiece vocabulary on our finanical corpora using the SentencePiece library. We produce both cased and uncased versions of `FinVocab`, with sizes of 28,573 and 30,873 tokens respectively. This is very similar to the 28,996 and 30,522 token sizes of the original BERT cased and uncased `BaseVocab`. 
- [FinVocab-Uncased](https://gohkust-my.sharepoint.com/:t:/g/personal/imyiyang_ust_hk/EchaAUzzYKhAidVhkqGp790BuA8UC5E9rTRhTmAnlGzZug?e=eniqml)
- [FinVocab-Cased](https://gohkust-my.sharepoint.com/:t:/g/personal/imyiyang_ust_hk/EX3C-KM9bTxOjdttsPslLZUBw_mh9Jdh8PB0WTv6b2tEIA?e=DYBVJY)

## Using FinBERT for financial sentiment classification

Finanical sentiment classification is a core NLP task in finance. FinBERT is shown to outperform vanilla BERT model on several financial sentiment classification task. Since FinBERT is in the same format as BERT, please refer to Google's BERT repo for downstream tasks. 

As a demostration, We provide a script for fine-tuning FinBERT for Finanical Phrase Bank dataset.
