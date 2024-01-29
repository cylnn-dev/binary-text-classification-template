import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from configs import model_name, training_args, device, tokenizer_config, path_to_save
from utility_functions import get_raw_csv, print_dataset_info, process_dataset, train, predict_test_set

if __name__ == '__main__':
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

    # optimizer = torch.optim.NAdam(model.parameters(), lr=1e-4, )
    optimizer = training_args['optimizer'](params=model.parameters())
    scheduler = training_args['scheduler'](optimizer=optimizer)
    accuracy_metric = training_args['accuracy_metric']
    f1_metric = training_args['f1_metric']  # for such an imbalanced dataset, accuracy is not a good metric

    # --------------------------

    # print('model summary:', model)

    data_collator = training_args['data_collator'](tokenizer=tokenizer)

    train_loader = DataLoader(train_set, batch_size=training_args['batch_size'], drop_last=True,
                              collate_fn=data_collator)
    valid_loader = DataLoader(valid_set, batch_size=training_args['batch_size'], drop_last=True,
                              collate_fn=data_collator)

    train(model, train_loader, valid_loader, optimizer, scheduler, criterion, path_to_save)

    # load the best model in terms of specific metric, this one uses f1_score for comparisons
    model.load_state_dict(torch.load(path_to_save))
    print('\n[INFO] The best model is loaded\n')

    # --- lastly, load the test dataset and print some predictions ---
    predict_test_set(model, data_collator, tokenizer, criterion)

