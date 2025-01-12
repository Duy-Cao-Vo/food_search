from transformers import BertConfig, BertModel
import torch
#
#  if model is on hugging face Hub
# model = BertModel.from_pretrained("bert-base-uncased")
# from local folder
model = BertModel.from_pretrained("model/")

from transformers import AutoModel, AutoTokenizer

phobert = AutoModel.from_pretrained("model")
tokenizer = AutoTokenizer.from_pretrained("model")

# INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!
sentence = 'Trà sữa'

input_ids = torch.tensor([tokenizer.encode(sentence)])

with torch.no_grad():
    features = phobert(input_ids)  # Models outputs are now tuples

print(features)