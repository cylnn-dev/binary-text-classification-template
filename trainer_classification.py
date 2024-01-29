import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, \
    Trainer, DefaultFlowCallback, EarlyStoppingCallback
from transformers.data import metrics
from functools import partial

from configs import huggingface_args, model_name, device, training_args, tokenizer_config, trainer_metrics
from utility_functions import get_raw_csv, print_dataset_info, process_dataset, compute_metrics, predict_test_set

if __name__ == '__main__':
    torch.cuda.empty_cache()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=training_args['num_labels'], ignore_mismatched_sizes=True).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = training_args['data_collator'](tokenizer=tokenizer)

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

    trainer_args = TrainingArguments(**huggingface_args)

    trainer = Trainer(
        model=model,
        args=trainer_args,
        data_collator=data_collator,
        train_dataset=train_set,  # do not forget to NOT convert it to torch's dataloader
        eval_dataset=valid_set,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, metrics=trainer_metrics),
        callbacks=[
            DefaultFlowCallback(),
            # ProgressCallback(),
            EarlyStoppingCallback(early_stopping_patience=10),
            # TensorBoardCallback(),  # todo: use this for good visualization
        ],
        # optimizers=(optimizer, get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5,
        # num_training_steps=len(train_loader)))

    )

    # start_warmup()

    print('=' * 30)
    print(f'training is beginning for {model_name}')
    print('=' * 30)

    trainer.train()  # todofixed: solve this checkpoint mess later -> do not use them at all

    # check if weights are changed! -> pass, weights are indeed changed for only the head
    # after_training_param = list(model.parameters())[-3]
    # print(f'{after_training_param=}')
    # print('are they equal? ->', torch.eq(before_training_param, after_training_param))

    criterion = nn.CrossEntropyLoss() if training_args['num_labels'] > 2 \
        else nn.BCEWithLogitsLoss(pos_weight=torch.tensor([train_negative_ratio], device=device))

    predict_test_set(model, data_collator, tokenizer, criterion)