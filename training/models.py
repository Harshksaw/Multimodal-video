import torch.nn as nn
from transformers import BertModel

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        
        for param in self.bert.parameters():
            param.requires_grad = False

        self.projection = nn.Linear(768, 128)
        
    def forward(self, input_ids, attention_mask):
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the last hidden state of the first token (CLS token) for classification
        pooler_output = outputs.pooler_output
        
        
        # Apply the projection layer
        return self.projection(pooler_output)