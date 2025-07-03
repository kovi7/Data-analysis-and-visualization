from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import torch

# on master node we do not use GPUs thus we switch to CPU
device_type = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#JUST ONCE TO DOWNLOAD THE MODEL FILE
pipeline_bfd = TokenClassificationPipeline( model=AutoModelForTokenClassification.from_pretrained("Rostlab/prot_bert_bfd_ss3"), tokenizer=AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd_ss3", skip_special_tokens=True), device=device_type)
