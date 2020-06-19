# FinBERT

`FinBERT` is a BERT model trained on financial communication text. The purpose is to enhance finaincal NLP research and practice. It is trained on the following three finanical communication corpus. The total corpora size is 4.9B tokens.

* Corporate Reports 10-K & 10-Q: 2.5B tokens 
* Earnings Call Transcripts: 1.3B tokens
* Analyst Reports: 1.1B tokens

`FinBERT` results in state-of-the-art performance on financial sentiment classification task, which is a core financial NLP task. 
With the release of `FinBERT`, we hope practitioners and researchers can utilize FinBERT for a wider range of applications where the prediction target goes beyond sentiment, such as financial-related outcomes including stock returns, stock volatilities, corporate fraud, etc.

## Download FinBERT

We provide four versions of pre-trained weights. 
- [FinBERT-FinVocab-Uncased](https://gohkust-my.sharepoint.com/:f:/g/personal/imyiyang_ust_hk/EksJcamJpclJlbMweFfB5DQB1XrsxURYN5GSqZw3jmSeSw?e=KAyhsX) (**Recommended**)
- [FinBERT-FinVocab-Cased](https://gohkust-my.sharepoint.com/:f:/g/personal/imyiyang_ust_hk/EgJZkmPlrdBLj6Kb4RXxwGwBymku6G-47QQrPYYDPJfr1Q?e=xA978z)
- FinBERT-BaseVocab-Uncased
- FinBERT-BaseVocab-Cased

`FinVocab` is a new WordPiece vocabulary on our finanical corpora using the SentencePiece library. We produce both cased and uncased versions of `FinVocab`, with sizes of 28,573 and 30,873 tokens respectively. This is very similar to the 28,996 and 30,522 token sizes of the original BERT cased and uncased `BaseVocab`. 
- [FinVocab-Uncased](https://gohkust-my.sharepoint.com/:t:/g/personal/imyiyang_ust_hk/EchaAUzzYKhAidVhkqGp790BuA8UC5E9rTRhTmAnlGzZug?e=eniqml)
- [FinVocab-Cased](https://gohkust-my.sharepoint.com/:t:/g/personal/imyiyang_ust_hk/EX3C-KM9bTxOjdttsPslLZUBw_mh9Jdh8PB0WTv6b2tEIA?e=DYBVJY)

## Using FinBERT for financial sentiment classification

Finanical sentiment classification is a core NLP task in finance. FinBERT is shown to outperform vanilla BERT model on several financial sentiment classification task. Since FinBERT is in the same format as BERT, please refer to Google's BERT repo for downstream tasks. 

As a demostration, We provide a script for fine-tuning FinBERT for Finanical Phrase Bank dataset.

In order to train FinBert on the Financial Phrase Bank dataset, please run the script as follows:

`python train_bert.py --cuda_device (cuda:device_id) --output_path (output directory) --vocab (vocab chosen)`\
 `--vocab_path (path to new vocab txt file) --data_dir (path to downloaded dataset) --weight_path (path to downloaded weights)`
 
There are 4 kinds of vocab to choose from: finance-cased, finance-uncased, base-cased, and base-uncased. 

*Note to run the script, one should first download the model weights, and the Financial Phrase Bank Dataset. 

### Downloading Financial Phrase Bank Dataset
The zip file for the Financial Phrase Bank Dataset has been provided for ease of download and use. 

### Environment:
To set up the evironment used to train and test the model, run `pip install -r requirements.txt`\
We would like to give special thanks to creators of pytorch_trained_bert (i.e. pytorch-transformers)


## Citation
    @misc{yang2020finbert,
        title={FinBERT: A Pretrained Language Model for Financial Communications},
        author={Yi Yang and Mark Christopher Siy UY and Allen Huang},
        year={2020},
        eprint={2006.08097},
        archivePrefix={arXiv},
        }

## Contact
Please post a Github issue or contact [imyiyang@ust.hk](imyiyang@ust.hk) if you have any questions.
