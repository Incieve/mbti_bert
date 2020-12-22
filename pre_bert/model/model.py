import torch.nn as nn
from transformers import BertModel


class PersonalityClassifier(nn.Module):

    def __init__(self, n_classes, bert_type):
        super(PersonalityClassifier, self).__init__()
        # Initialize pretrained BERT model -> Model is freezed
        # i.e. weights of the model are constants and are not trainable
        self.bert = BertModel.from_pretrained(bert_type)
        # Define dropout
        self.drop = nn.Dropout(p=0.1)
        # Define last linear layer after BERT model -> this is what we are training
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        # Returns output from BERT based on input and mask
        _, pooled_output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
        )
        # Perform dropout on the output of BERT
        output = self.drop(pooled_output)
        # Returns output after linear layer dim(self.out(output)) == n_clases
        # So basically these are our predictions
        return self.out(output)