# FinBERT

**\*\*\*\*\* June 2, 2022: More fine-tuned FinBERT models available\*\*\*\*\***

Visit [FinBERT.AI](https://finbert.ai/) for more details on the recent development of FinBERT.

We have fine-tuned FinBERT pretrained model on several financial NLP tasks, all outperforming traditional machine learning models, deep learning models, and fine-tuned BERT models. All the fine-tuned FinBERT models are publicly hosted at Huggingface 🤗. Specifically, we have the following:

- **FinBERT-Pretrained**: The pretrained FinBERT model on large-scale financial text. [link](https://huggingface.co/yiyanghkust/finbert-pretrain)
- **FinBERT-Sentiment**: for sentiment classification task. [link](https://huggingface.co/yiyanghkust/finbert-tone)
- **FinBERT-ESG**: for ESG classification task. [link](https://huggingface.co/yiyanghkust/finbert-esg)
- **FinBERT-FLS**: for forward-looking statement (FLS) classification task. [link](https://huggingface.co/yiyanghkust/finbert-fls)

In this Github repo, 
- [FinBERT-demo.ipynb](https://github.com/yya518/FinBERT/blob/master/FinBERT-demo.ipynb) demonstrates how to apply fine-tuned FinBERT model on specific NLP tasks. 
- [finetune.ipynb](https://github.com/yya518/FinBERT/blob/master/finetune.ipynb) illustrates the process of fine-tuning FinBERT.


**Background**: 

`FinBERT` is a BERT model pre-trained on financial communication text. The purpose is to enhance finaincal NLP research and practice. It is trained on the following three finanical communication corpus. The total corpora size is 4.9B tokens.

* Corporate Reports 10-K & 10-Q: 2.5B tokens 
* Earnings Call Transcripts: 1.3B tokens
* Analyst Reports: 1.1B tokens

`FinBERT` results in state-of-the-art performance on various financial NLP task, including sentiment analysis, ESG classification, forward-looking statement (FLS) classification. With the release of `FinBERT`, we hope practitioners and researchers can utilize `FinBERT` for a wider range of applications where the prediction target goes beyond sentiment, such as financial-related outcomes including stock returns, stock volatilities, corporate fraud, etc.

---
**\*\*\*\*\* July 30, 2021: migrated to Huggingface 🤗\*\*\*\*\***

The fine-tuned `FinBERT` model for financial sentiment classification has been uploaded and integrated with Huggingface's [`transformers`](https://huggingface.co/transformers/) library. This model is fine-tuned on 10,000 manually annotated (positive, negative, neutral) sentences from analyst reports. This model achieves superior performance on financial tone anlaysis task. If you are simply interested in using `FinBERT` for financial tone analysis, give it a try.

```javascript
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

sentences = ["there is a shortage of capital, and we need extra financing", 
             "growth is strong and we have plenty of liquidity", 
             "there are doubts about our finances", 
             "profits are flat"]

inputs = tokenizer(sentences, return_tensors="pt", padding=True)
outputs = finbert(**inputs)[0]

labels = {0:'neutral', 1:'positive',2:'negative'}
for idx, sent in enumerate(sentences):
    print(sent, '----', labels[np.argmax(outputs.detach().numpy()[idx])])
    
'''
there is a shortage of capital, and we need extra financing ---- negative
growth is strong and we have plenty of liquidity ---- positive
there are doubts about our finances ---- negative
profits are flat ---- neutral
'''
    
```

***

**\*\*\*\*\* Jun 16, 2020: Pretrained FinBERT Model Released\*\*\*\*\***

We provide four versions of pre-trained FinBERT weights. 
- [FinBERT-FinVocab-Uncased](https://gohkust-my.sharepoint.com/:f:/g/personal/imyiyang_ust_hk/EksJcamJpclJlbMweFfB5DQB1XrsxURYN5GSqZw3jmSeSw?e=KAyhsX) (**Recommended**)
- [FinBERT-FinVocab-Cased](https://gohkust-my.sharepoint.com/:f:/g/personal/imyiyang_ust_hk/EgJZkmPlrdBLj6Kb4RXxwGwBymku6G-47QQrPYYDPJfr1Q?e=xA978z)
- [FinBERT-BaseVocab-Uncased](https://gohkust-my.sharepoint.com/:f:/g/personal/imyiyang_ust_hk/ErcYR77ZaxBAnQsmjIJF5joBapAf0HDaH0vWr_WXnoF1sA?e=oBTsSk)
- [FinBERT-BaseVocab-Cased](https://gohkust-my.sharepoint.com/:f:/g/personal/imyiyang_ust_hk/EtBK8m0MBC1Np5sAN-s5ZHsBW2dGCfBvoZtXyD_Xa9ywGw?e=h3veaz)

`FinVocab` is a new WordPiece vocabulary on our finanical corpora using the SentencePiece library. We produce both cased and uncased versions of `FinVocab`, with sizes of 28,573 and 30,873 tokens respectively. This is very similar to the 28,996 and 30,522 token sizes of the original BERT cased and uncased `BaseVocab`. 
- [FinVocab-Uncased](https://gohkust-my.sharepoint.com/:t:/g/personal/imyiyang_ust_hk/EX3C-KM9bTxOjdttsPslLZUBw_mh9Jdh8PB0WTv6b2tEIA?e=DYBVJY)
- [FinVocab-Cased](https://gohkust-my.sharepoint.com/:t:/g/personal/imyiyang_ust_hk/EchaAUzzYKhAidVhkqGp790BuA8UC5E9rTRhTmAnlGzZug?e=eniqml)

 

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
