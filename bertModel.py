from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertConfig

class BertClassification(nn.Module):
   
    def __init__(self, weight_path, num_labels=2, vocab="base-cased"):
        super(BertClassification, self).__init__()
        self.num_labels = num_labels
        self.vocab = vocab 
        if self.vocab == "base-cased":
            self.bert = BertModel.from_pretrained(weight_path)
            self.config = BertConfig(vocab_size_or_config_json_file=28996, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

        elif self.vocab == "base-uncased":
            self.bert = BertModel.from_pretrained(weight_path)
            self.config = BertConfig(vocab_size_or_config_json_file=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
            
        elif self.vocab == "finance-cased":
            self.bert = BertModel.from_pretrained(weight_path)
            self.config = BertConfig(vocab_size_or_config_json_file=28573, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

        elif self.vocab =="finance-uncased":
            self.bert = BertModel.from_pretrained(weight_path)
            self.config = BertConfig(vocab_size_or_config_json_file=30873, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        nn.init.xavier_normal(self.classifier.weight)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, graphEmbeddings=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
       
        logits = self.classifier(pooled_output)
            
        return logits

class dense_opt():
    def __init__(self, model):
        super(dense_opt, self).__init__()
        self.lrlast = .001
        self.lrmain = .00001
        self.optim = optim.Adam(
        [ {"params":model.bert.parameters(),"lr": self.lrmain},
          {"params":model.classifier.parameters(), "lr": self.lrlast},
       ])
    
    def get_optim(self):
        return self.optim