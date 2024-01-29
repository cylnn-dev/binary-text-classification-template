
from neural_compressor.utils.pytorch import load
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from configs import model_name, training_args, device
from utility_functions import predict_test_set

model_path = fr'quantized_models/{model_name}/best_model.pt'

model = AutoModelForSequenceClassification.from_pretrained(
 model_name, num_labels=training_args['num_labels'], ignore_mismatched_sizes=True).to(device)


q_model = load(model_path, model).to(device)


tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = training_args['data_collator'](tokenizer=tokenizer)

criterion = nn.CrossEntropyLoss()

print('=' * 30)
print('results of quantized model')
print('=' * 30)
predict_test_set(q_model, data_collator, tokenizer, criterion, sample_size=20)


print('=' * 30)
print('results of previous model')
print('=' * 30)