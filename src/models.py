import torch.nn as nn
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification,
)

#def get_model(config):
#    if config.model == "bert":
#        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
#        model = BertForSequenceClassification.from_pretrained(
#            "bert-base-multilingual-cased", num_labels=3
#        )
#    else:
#        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
#        model = RobertaForSequenceClassification.from_pretrained(
#            "roberta-large", num_labels=3
#        )
#    return tokenizer, model

def get_model(model_type, num_labels):
    if model_type == "bert":
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased", num_labels=num_labels
        )
    elif model_type=="roberta":
        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-large", num_labels=num_labels
        )
    return model


def get_tokenizer(model_type):
    if model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    elif model_type=="roberta":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    return tokenizer



class LangModel(nn.Module):
    def __init__(self, num_labels=3, model_cls = BertForSequenceClassification, tokenizer_cls = BertTokenizer, 
                                   model_subtype='bert-base-multilingual-cased', tokenizer_subtype='bert-base-cased'):
        super(LangModel, self).__init__()
        #if args.model == "bert":
        #    self.lm = BertForSequenceClassification.from_pretrained(
        #        "bert-base-multilingual-cased", num_labels=args.num_labels
        #    )
        #    self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        #elif args.model=="roberta":
        #    self.lm = RobertaForSequenceClassification.from_pretrained(
        #        "roberta-large", num_labels=args.num_labels
        #    )
        #    self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        self.lm = model_cls.from_pretrained(model_subtype, num_labels=num_labels)
        self.tokenizer = tokenizer_cls.from_pretrained(tokenizer_subtype)

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, **kwargs):
        return self.lm(**kwargs)       

class Base:
    base = LangModel
    args = list()
    kwargs = {}

class Bert(Base):
    kwargs = {'num_labels': 3, 'model_cls': BertForSequenceClassification, 'tokenizer_cls': BertTokenizer, 
              'model_subtype': 'bert-base-multilingual-cased', 'tokenizer_subtype': 'bert-base-cased'}

class Roberta(Base):
    kwargs = {'num_labels': 3, 'model_cls': RobertaForSequenceClassification,  'tokenizer_cls': RobertaTokenizer, 
              'model_subtype': 'roberta-large', 'tokenizer_subtype': 'roberta-large'}
