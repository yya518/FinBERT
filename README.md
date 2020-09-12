# FinBERT

`FinBERT` is a BERT model pre-trained on financial communication text. The purpose is to enhance finaincal NLP research and practice. It is trained on the following three finanical communication corpus. The total corpora size is 4.9B tokens.

* Corporate Reports 10-K & 10-Q: 2.5B tokens 
* Earnings Call Transcripts: 1.3B tokens
* Analyst Reports: 1.1B tokens

`FinBERT` results in state-of-the-art performance on financial sentiment classification task, which is a core financial NLP task. 
With the release of `FinBERT`, we hope practitioners and researchers can utilize `FinBERT` for a wider range of applications where the prediction target goes beyond sentiment, such as financial-related outcomes including stock returns, stock volatilities, corporate fraud, etc.

**You can use FinBERT in two ways:**
1. Pre-trained model. You can fine-tuned FinBERT with your own dataset. FinBERT is most suitable for financial NLP tasks. We have the provided several FinBERT models in below, as well as the fine-tune scripts.
2. Fine-tuned model. If you are interested in simply using FinBERT for financial sentiment classification prediction, we provide a fine-tuned FinBERT model that is fine tuned on 10,000 manually annotated analyst statements. This dataset has been used in Accounting literature for analyst tone analysis (Huang et al., The Accounting Review, 2014).

# 1. Pre-trained model
### Download FinBERT

We provide four versions of pre-trained weights. 
- [FinBERT-FinVocab-Uncased](https://gohkust-my.sharepoint.com/:f:/g/personal/imyiyang_ust_hk/EksJcamJpclJlbMweFfB5DQB1XrsxURYN5GSqZw3jmSeSw?e=KAyhsX) (**Recommended**)
- [FinBERT-FinVocab-Cased](https://gohkust-my.sharepoint.com/:f:/g/personal/imyiyang_ust_hk/EgJZkmPlrdBLj6Kb4RXxwGwBymku6G-47QQrPYYDPJfr1Q?e=xA978z)
- [FinBERT-BaseVocab-Uncased](https://gohkust-my.sharepoint.com/:f:/g/personal/imyiyang_ust_hk/ErcYR77ZaxBAnQsmjIJF5joBapAf0HDaH0vWr_WXnoF1sA?e=oBTsSk)
- [FinBERT-BaseVocab-Cased](https://gohkust-my.sharepoint.com/:f:/g/personal/imyiyang_ust_hk/EtBK8m0MBC1Np5sAN-s5ZHsBW2dGCfBvoZtXyD_Xa9ywGw?e=h3veaz)

`FinVocab` is a new WordPiece vocabulary on our finanical corpora using the SentencePiece library. We produce both cased and uncased versions of `FinVocab`, with sizes of 28,573 and 30,873 tokens respectively. This is very similar to the 28,996 and 30,522 token sizes of the original BERT cased and uncased `BaseVocab`. 
- [FinVocab-Uncased](https://gohkust-my.sharepoint.com/:t:/g/personal/imyiyang_ust_hk/EX3C-KM9bTxOjdttsPslLZUBw_mh9Jdh8PB0WTv6b2tEIA?e=DYBVJY)
- [FinVocab-Cased](https://gohkust-my.sharepoint.com/:t:/g/personal/imyiyang_ust_hk/EchaAUzzYKhAidVhkqGp790BuA8UC5E9rTRhTmAnlGzZug?e=eniqml)

#### Downloading Financial Phrase Bank Dataset
The datase is collected by [Malo et al. 2014](https://arxiv.org/abs/1307.5336), and can be downloaded from [this link](https://www.researchgate.net/profile/Pekka_Malo/publication/251231364_FinancialPhraseBank-v10/data/0c96051eee4fb1d56e000000/FinancialPhraseBank-v10.zip?origin=publication_list). The zip file for the Financial Phrase Bank Dataset has been provided for ease of download and use. 

#### Environment:
To set up the evironment used to train and test the model, run `pip install -r requirements.txt`\
We would like to give special thanks to the creators of pytorch-pretrained-bert (i.e. pytorch-transformers)

In order to fine-tune `FinBERT` on the Financial Phrase Bank dataset, please run the script as follows:

`python train_bert.py --cuda_device (cuda:device_id) --output_path (output directory) --vocab (vocab chosen)`\
 `--vocab_path (path to new vocab txt file) --data_dir (path to downloaded dataset) --weight_path (path to downloaded weights)`
 
There are 4 kinds of vocab to choose from: `FinVocab-Uncased`, `FinVocab-Cased`, and Google's BERT Base-Uncased and Base-Cased. 

*Note to run the script, one should first download the model weights, and the Financial Phrase Bank Dataset.*

# 2. Fine-tuned model
### Using FinBERT for financial sentiment classification
If you are simply interested in using FinBERT for downstream sentiment classification task, we have a fine-tuned FinBERT for your use. This fine-tuned FinBERT model is fine-tuned on 10,000 analyst statements for tone prediction task (positive, negative, neutral). We provide a [Jupyter notebook](https://github.com/yya518/FinBERT/blob/master/FinBert%20Model%20Example.ipynb) to show how you can use it with your own data. For comparison purpose, we also provided a pre-trained Naive Bayes Model.  The fine-tuned FinBERT has significantly better performance than the Naive Bayes model, and it can gauge finanical text tone with high accuracy.


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
