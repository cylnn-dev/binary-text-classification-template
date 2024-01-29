from functools import partial

import evaluate
import torch
from neural_compressor.config import TuningCriterion, AccuracyCriterion, PostTrainingQuantConfig
from neural_compressor.quantization import fit
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from configs import device, model_name, training_args, tokenizer_config, path_to_save
from utility_functions import process_dataset, print_dataset_info, get_raw_csv, validate

torch.cuda.empty_cache()

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=training_args['num_labels'], ignore_mismatched_sizes=True).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# --- dataset settings ---

train_set_raw, valid_set_raw = get_raw_csv(training_args['train_filepath'],
                                           reduce_ratio=training_args['reduce_ratio'],
                                           valid_size=training_args['valid_size'],
                                           return_valid=training_args['return_valid'])
y_train = train_set_raw['label']
y_valid = valid_set_raw['label']
train_negative_ratio = y_train[y_train == 0].count() / y_train[y_train == 1].count()
print_dataset_info(y_train, y_valid)

train_set = process_dataset(train_set_raw, tokenizer, tokenizer_config)
valid_set = process_dataset(valid_set_raw, tokenizer, tokenizer_config)
# --------------------------

# --- training arguments ---
# https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
criterion = nn.CrossEntropyLoss() if training_args['num_labels'] > 2 else nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([train_negative_ratio], device=device))

data_collator = training_args['data_collator'](tokenizer=tokenizer)

train_loader = DataLoader(train_set, batch_size=training_args['batch_size'], drop_last=True,
                          collate_fn=data_collator)
valid_loader = DataLoader(valid_set, batch_size=training_args['batch_size'], drop_last=True,
                          collate_fn=data_collator)

# post quantization using intel compressor
tuning_criterion = TuningCriterion(
    strategy='basic',
    max_trials=5,
    timeout=0,  # early stopping
)

accuracy_criterion = AccuracyCriterion(criterion='relative', tolerable_loss=0.01)

intel_config = PostTrainingQuantConfig(
    device='cpu',  # todo: i don't know why, but it works only in cpu, maybe this specific gpu is not supported
    approach='static',  # or use 'dynamic'
    tuning_criterion=tuning_criterion,
    accuracy_criterion=accuracy_criterion,
)

model.load_state_dict(torch.load(path_to_save))  # this model is trained before, see checkpoints/
print('\n[INFO] The best model is loaded\n')

accuracy_metric = evaluate.load('accuracy')
f1_metric = evaluate.load("f1")  # for such an imbalanced dataset, accuracy is not a good metric

q_model = fit(model=model, conf=intel_config, calib_dataloader=train_loader,
              # eval_func=partial(eval_func, dataloader=valid_loader),
              eval_func=partial(validate, criterion=criterion, dataloader=valid_loader, best_val_f1=0.0, ),
              )
# warning: do not give relative path. use the full path, otherwise go to your C driver
q_model.save(fr'/quantized_models/{model_name}')
