from functools import partial
from pathlib import Path

import evaluate
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import DataCollatorWithPadding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

tokenizer_config = dict(
    truncation=True,
    padding='max_length',
    max_length=32,
)

# --- choose a model from the list, other models (similar ones) will probably work out of box ---
# 'distilbert-base-uncased'
# 'siebert/sentiment-roberta-large-english'
# 'nlptown/bert-base-multilingual-uncased-sentiment'
model_name = 'distilbert-base-uncased'
saving_name = model_name.replace('/', '-').replace('\\', '-').lower().strip()
path_to_save = Path(fr'checkpoints/{saving_name}.pt')

# trainer_classification.py uses some elements of the training_args, but not all of them
# this can be mitigated by collapsing them into one dictionary, but trainer is just a demonstration
# if possible, please use manual_classification
training_args = dict(
    path_to_save=Path(fr'checkpoints/{saving_name}.pt'),
    epochs=10,
    batch_size=96,
    num_labels=2,  # try out num_labels=2, automatically chooses negative and positive ones.
    optimizer=partial(torch.optim.NAdam, lr=1e-5),
    # optimizer=partial(torch.optim.AdamW, lr=1e-5),
    # choosing criterion in the main script would be better for code readability
    scheduler=partial(torch.optim.lr_scheduler.StepLR, step_size=2, gamma=0.95),
    accuracy_metric=evaluate.load('accuracy'),
    f1_metric=evaluate.load("f1", average='macro'),
    # for such an imbalanced dataset, accuracy is not a good metric
    # average='macro' probably not working and throws no error, too! Implement so f1_metric.compute(.., average='macro')
    data_collator=DataCollatorWithPadding,
    train_filepath=r'datasets/imdb_binary/test.csv',
    test_filepath=r'datasets/imdb_binary/train.csv',
    sentiment_mapping={'positive': 1, 'negative': 0},
    reduce_ratio=None,  # use only %reduce_ratio of the dataset, 0.1 -> %10 of train_set
    valid_size=0.2,
    return_valid=True,
    writer=SummaryWriter()  # turn off tensorboard by setting -> writer=None
    # how to see tensorboard results -> open terminal in virtual env -> tensorboard --logdir=runs
)

huggingface_args = dict(
    output_dir='trainer_stars/trainer_cp',
    overwrite_output_dir=True,
    evaluation_strategy='epoch',  # epoch
    per_device_train_batch_size=training_args['batch_size'],
    per_device_eval_batch_size=training_args['batch_size'],
    # auto_find_batch_size=True, # do not trust this variable. Experimenting seems a good way
    learning_rate=5e-5,
    num_train_epochs=training_args['epochs'],
    lr_scheduler_type='linear',
    # warmup_steps=5,
    log_level='debug',
    logging_strategy='epoch',
    logging_nan_inf_filter=True,
    save_strategy='epoch',
    save_steps=500,
    save_total_limit=2,  # save only the last epoch and the best model
    # optimization variables, test them for specific hardware
    # if all of them false: circa 1.15 minutes (%1 of set)
    no_cuda=False,
    bf16=False,
    fp16=False,
    tf32=False,
    #################
    dataloader_drop_last=True,
    eval_steps=1,
    dataloader_num_workers=0,  # os.cpu_count() is way slower if using fast tokenizer.
    load_best_model_at_end=True,
    metric_for_best_model='eval_f1_metric',  # 'f1' ?
    greater_is_better=True,
    deepspeed=None,  # todo: try out this deepspeed by showing ds_config.json
    debug='underflow_overflow',
    # detects overflow in model’s input/outputs and reports the last frames that led to the event
    optim='adamw_hf',
    # possible values: adamw_hf, adamw_torch, adamw_torch_fused, adamw_apex_fused, adamw_anyprecision or adafactor.
    group_by_length=False,  # useful if padding is dynamic
    gradient_checkpointing=False,
    # If True, use gradient checkpointing to save memory at the expense of slower backward pass.
)

trainer_metrics = [('accuracy', training_args['accuracy_metric']), ('f1_metric', training_args['f1_metric'])]
