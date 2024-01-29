import datetime
import subprocess
import timeit
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm.auto
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from torch.utils.data import DataLoader
from transformers import EarlyStoppingCallback

from configs import device, training_args, tokenizer_config, model_name
from datasets import Dataset


def perf_timer(code_to_measure, n_executions=2):
    time_measure = timeit.timeit(lambda: exec(code_to_measure), number=n_executions)
    print('time_measure:', time_measure / n_executions)


def get_gpu_util():
    """
    Fetches GPU utilization percentage using 'nvidia-smi' command.

    Returns:
        int: GPU utilization percentage.

    Example:
        >>> get_gpu_util()
        75
    """
    cmd = 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader'
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
    gpu_util = int(result.stdout.decode().split()[0])
    return gpu_util


def preprocess(samples, tokenizer, tokenizer_config: dict = None):
    # tokenizer configs: truncation=True, padding='max_length',max_length=10,
    return tokenizer(samples['text'], **tokenizer_config)


def return_optimizer(optimizer, model):
    return partial(optimizer, params=model.parameters())


def process_dataset(raw_dataset: pd.DataFrame, tokenizer, tokenizer_config: dict = None,
                    return_text_column: bool = False, num_proc: int = 1):
    """
    Args:
        raw_dataset (pd.DataFrame): dataset which contains (value, text) pairs in DataFrame format
        tokenizer: Tokenizer specific to the HuggingFace model.
        tokenizer_config (dict, optional): Additional configuration for the tokenizer.
        return_text_column (bool, optional): If True, returns the tokenized texts along with the processed dataset.
        num_proc (int, optional): Number of processes to use for parallel processing.

    Returns:
        - If return_text_column is False:
            - tokenized_dataset (datasets.Dataset): Processed and tokenized dataset with 'text' column removed,
              formatted for PyTorch.
        - If return_text_column is True:
            - tokenized_texts (datasets.DatasetColumn): Tokenized 'text' column.
            - tokenized_dataset (datasets.Dataset): Processed and tokenized dataset with 'text' column removed,
              formatted for PyTorch.
    """
    # this method is 3x slower than the process_dataset(). Currently, the best method is converting dataframe to dataset
    # tokenized_texts = tokenizer(train_df['text'].tolist(), truncation=True, padding='max_length')

    # test data has +1000 duplicates like  'thank you'
    # to see how many duplicated item in a dataframe w.r.t a specific column:
    # test_set_raw.duplicated(subset=['text']).sum()
    raw_dataset = raw_dataset.drop_duplicates(subset=['text'], keep='last')
    tokenized_dataset = Dataset.from_pandas(raw_dataset).remove_columns('__index_level_0__').shuffle(
        writer_batch_size=10_000)
    tokenized_dataset = tokenized_dataset.map(
        partial(preprocess, tokenizer=tokenizer, tokenizer_config=tokenizer_config), num_proc=num_proc, batched=True
    )

    if not return_text_column:
        return tokenized_dataset.remove_columns('text').with_format('torch')
    else:
        return tokenized_dataset['text'], tokenized_dataset.remove_columns('text').with_format('torch')


def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, epoch_index):
    """
    Train the model for one epoch using the provided training data.

    Args:
        model: The PyTorch model to be trained.
        train_loader: DataLoader containing training data.
        optimizer: The optimizer used for training.
        scheduler: Learning rate scheduler.
        criterion: The loss criterion used for optimization.
        epoch_index (int): The index of the current epoch.

    Returns:
         None
    """
    calculate_loss = calculate_loss_for_multi if training_args['num_labels'] > 2 else calculate_loss_for_binary
    dataloader_prog = tqdm.tqdm(train_loader, position=0, leave=True)
    model.train()

    running_loss = 0.0
    for i, batch in enumerate(dataloader_prog):
        input_ids, mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(
            device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=mask).logits
        loss: torch.Tensor = calculate_loss(labels, outputs, criterion)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        dataloader_prog.set_description(
            f'epoch: {epoch_index + 1} -> loss: {loss.item():.3e}, lr: {scheduler.get_last_lr()[0]:.3e}, '
            f'gpu_util: %{get_gpu_util()} ::')

        predicted_labels = torch.argmax(outputs, dim=1)
        training_args['accuracy_metric'].add_batch(predictions=predicted_labels, references=labels)
        training_args['f1_metric'].add_batch(predictions=predicted_labels, references=labels)

    scheduler.step()
    epoch_loss = running_loss / len(train_loader)  # summed losses during epoch / number of batches
    train_acc = training_args["accuracy_metric"].compute()["accuracy"]
    train_f1 = training_args["f1_metric"].compute(average="macro")["f1"]
    print(
        f'epoch: {epoch_index + 1} -> '
        f'train_loss: {epoch_loss: .3e}, '
        f'train_acc: {train_acc: .3f}, '
        f'train_f1: {train_f1: .3f}')

    if training_args['writer'] is not None:
        training_args['writer'].add_scalar('loss/train', epoch_loss, global_step=epoch_index + 1)
        training_args['writer'].add_scalar('lr/train', scheduler.get_last_lr()[0], global_step=epoch_index + 1)
        training_args['writer'].add_scalar('gpu_util/train', get_gpu_util(), global_step=epoch_index + 1)
        training_args['writer'].add_scalar('train_acc/train', train_acc, global_step=epoch_index + 1)
        training_args['writer'].add_scalar('train_f1/train', train_f1, global_step=epoch_index + 1)


def train(model, train_loader, valid_loader, optimizer, scheduler, criterion, path_to_save):
    """
    Train a PyTorch model using the provided training and validation data.

    Args:
        model: The PyTorch model to be trained.
        train_loader: DataLoader containing training data.
        valid_loader: DataLoader containing validation data.
        optimizer: The optimizer used for training.
        scheduler: Learning rate scheduler.
        criterion: The loss criterion used for optimization.
        path_to_save (str): Path to save the trained model.

    Returns:
        None

    Notes:
        This function trains the model for the specified number of epochs, calling train_one_epoch for each epoch.
        It performs validation after each epoch using the validate function.
        The best model based on validation F1 score is saved to the specified path.
    """
    torch.cuda.empty_cache()
    timestamp = datetime.datetime.now().strftime("%d.%m.%Y, %H:%M:%S")

    best_val_f1 = 0.0
    print(f'\n\nTraining begins for {training_args["epochs"]=}, {model_name=}, {device=} date: {timestamp}')
    for epoch in range(training_args["epochs"]):
        train_one_epoch(model, train_loader, optimizer, scheduler, criterion, epoch_index=epoch)
        best_val_f1 = validate(model, criterion, valid_loader, best_val_f1, epoch, path_to_save, mode='val')

    # except torch.cuda.OutOfMemoryError:
    #     print('[Warning] not enough Vram for our model, fall back to cpu!')
    #     device = torch.device('cpu')
    #     model.to(device)
    #     torch.cuda.empty_cache()
    #     timestamp = datetime.datetime.now().strftime("%d.%m.%Y, %H:%M:%S")
    #     best_val_f1 = 0.0
    #     print(f'\n\nTraining begins for {epochs=}, {model_name=}, {device=} date: {timestamp}')
    #     for epoch in range(epochs):
    #         train_one_epoch(epoch, train_loader, model)
    #         validate(valid_loader, best_val_f1, num_classes, epoch, path_to_save)


@torch.inference_mode()
def validate(model, criterion, dataloader: DataLoader, best_val_f1, epoch_index: int = 0, path_to_save: Path = None,
             mode: str = 'valid'):
    """
    Validate a PyTorch model using the provided validation data.

    Args:
        model: The PyTorch model to be validated.
        criterion: The loss criterion used for evaluation.
        dataloader: DataLoader containing validation data.
        best_val_f1: Best F1 score achieved during validation.
        epoch_index (int, optional): The index of the current epoch.
        path_to_save (Path, optional): Path to save the best model checkpoint.
        mode (str, optional): Mode of validation ('valid' or 'test').

    Returns:
        float: Updated the best validation F1 score.

    Notes:
        This function assumes binary classification for num_labels <= 2 and multi-class classification otherwise.
        Validation progress is displayed using tqdm.
    """
    if path_to_save is not None:
        root_directory = Path(str(path_to_save).split('\\')[0])
        if not root_directory.exists():
            root_directory.mkdir()

    calculate_loss = calculate_loss_for_multi if training_args['num_labels'] > 2 else calculate_loss_for_binary
    dataloader_prog = tqdm.tqdm(dataloader, position=0, leave=True)
    model.eval()

    running_valid_loss = 0.0
    for i, batch in enumerate(dataloader_prog):
        input_ids, mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(
            device)
        outputs = model(input_ids.to(device), mask.to(device)).logits
        loss: torch.Tensor = calculate_loss(labels, outputs, criterion)
        running_valid_loss += loss.item()

        predicted_labels = torch.argmax(outputs, dim=1)
        training_args['accuracy_metric'].add_batch(predictions=predicted_labels, references=labels)
        training_args['f1_metric'].add_batch(predictions=predicted_labels, references=labels)

        dataloader_prog.set_description('evaluation continues' + '.' * (i % 4))

    epoch_loss_valid = running_valid_loss / len(dataloader)  # summed losses during epoch / number of batches
    valid_accuracy = training_args['accuracy_metric'].compute()["accuracy"]
    valid_f1 = training_args['f1_metric'].compute(average='macro')['f1']
    print(
        f'epoch: {epoch_index + 1} -> '
        f'{mode}_loss: {epoch_loss_valid: .3e}, '
        f'{mode}_acc: {valid_accuracy: .3f}, '
        f'{mode}_f1: {valid_f1: .3f}',
    )

    if training_args['writer'] is not None:
        training_args['writer'].add_scalar(f'loss/{mode}', epoch_loss_valid, global_step=epoch_index + 1)
        training_args['writer'].add_scalar(f'{mode}_acc/{mode}', valid_accuracy, global_step=epoch_index + 1)
        training_args['writer'].add_scalar(f'{mode}_f1/{mode}', valid_f1, global_step=epoch_index + 1)

    if best_val_f1 < valid_f1:
        best_val_f1 = valid_f1
        print(f'\t--> current_best_{mode}_f1_score:{best_val_f1: .4f}')
        if path_to_save is not None:
            torch.save(model.state_dict(), path_to_save)
            print(f'model is saved successfully! See file: {path_to_save.resolve()}')
    return best_val_f1


def calculate_loss_for_binary(labels, outputs, criterion):
    return criterion(outputs, torch.eye(training_args['num_labels'], device=device)[labels])


def calculate_loss_for_multi(labels, outputs, criterion):
    return criterion(outputs, labels)


def get_raw_csv(path_to_dataset: str, reduce_ratio: float = None, valid_size: float = 0.3, return_valid: bool = True):
    """
    Load and preprocess a CSV dataset from the given path.

    Args:
        path_to_dataset (str): Path to the CSV dataset file.
        reduce_ratio (float, optional): Ratio to reduce the dataset size for limited resources.
        valid_size (float, optional): Proportion of the dataset to use for validation.
        return_valid (bool, optional): If True, returns both training and validation datasets.

    Returns:
        pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]: Processed DataFrame(s) containing (text, label) pairs.
            If return_valid is True, returns a tuple of (train_df, valid_df), else returns only train_df.

    Notes:
        - The function reads the CSV file into a DataFrame and performs column renaming.
        - Optionally maps sentiment labels to numerical values based on a provided mapping.
        - If reduce_ratio is specified, reduces the dataset size for resource-constrained environments.
        - Prints the count of each unique label in the dataset.
        - Optionally returns both training and validation datasets using train_test_split.
    """
    df = pd.read_csv(path_to_dataset)

    if training_args['sentiment_mapping'] is not None:
        df['label'] = df['sentiment'].map(training_args['sentiment_mapping'])
        df.drop(columns=['sentiment'], inplace=True)

    df.rename(columns={"review": "text"}, inplace=True)

    # reduce the dataset to work on an ultralight laptop with 2 GB of Vram
    if reduce_ratio is not None:
        df, _ = train_test_split(df, train_size=reduce_ratio, shuffle=True, stratify=df['label'])
    print(df.head())

    # print(f'label 0 count: {(df.label.values == 0).sum()}\nlabel 1 count: {(df.label.values == 1).sum()} ')
    unique_labels = sorted(df['label'].unique().tolist())
    for label in unique_labels:
        count = (df.label.values == label).sum()
        print(f"label {label} count: {count}")
    if return_valid:
        return train_test_split(df, test_size=valid_size, shuffle=True, stratify=df['label'])
    else:
        return df


def print_dataset_info(y_train, y_valid):
    """
    A simple function to print information about the labels in training and validation datasets.

    Args:
        y_train: Labels of the training dataset.
        y_valid: Labels of the validation dataset.

    Returns:
        None

    Notes:
        - Calls the print_label_statistics function for both training and validation datasets.
    """
    unique_labels = sorted(y_train.unique().tolist())
    print_label_statistics(unique_labels, y_train)
    print_label_statistics(unique_labels, y_valid)


def print_label_statistics(unique_labels, label_set):
    """
    Print information about the distribution of labels in a dataset.

    Args:
        unique_labels: List of unique labels present in the dataset.
        label_set: Labels of the dataset.

    Returns:
        None

    Notes:
        - Calculates and prints the count and percentage distribution of each unique label in the dataset. It was used
        to see how imbalanced the data is.
    """
    len_set = len(label_set)
    print(f'samples in train_data: {len_set}')
    for label in unique_labels:
        label_size = label_set[label_set == label].count()
        print(f'\t-> label {label}: {label_size},\t %{label_size / len_set * 100: .3f}')


@torch.inference_mode()
def predict_test_set(model, data_collator, tokenizer, criterion, sample_size=20):
    """
    Make predictions on the test set using a trained PyTorch model.

    Args:
        model: The PyTorch model for making predictions.
        data_collator: Data collator for creating batches from the test set.
        tokenizer: Tokenizer for processing the test set.
        criterion: Loss criterion used for evaluation.
        sample_size (int, optional): Number of samples to display predictions for.

    Returns:
        None

    Notes:
        - The function loads the raw test set, processes it, and creates a DataLoader for batch processing.
        - Calls the validate function to print accuracy and F1 scores on the test set.
        - Extracts a sample from the test set and displays the predicted labels alongside true labels.
    """
    print('get scores for test_data\n')
    test_set_raw = get_raw_csv(training_args['test_filepath'], return_valid=False)
    test_set_texts, test_set = process_dataset(test_set_raw, tokenizer, tokenizer_config, return_text_column=True)

    test_loader = DataLoader(test_set, batch_size=training_args['batch_size'], drop_last=True, collate_fn=data_collator)

    best_test_f1 = 0.0
    validate(model, criterion, test_loader, best_test_f1, mode='test')  # to see accuracy and f1 scores

    sample_set: dict = test_set[:sample_size]
    sample_texts = test_set_texts[:sample_size]

    model.eval()

    pred_probas = model(sample_set['input_ids'].to(device), sample_set['attention_mask'].to(device)).logits
    pred_labels = torch.argmax(pred_probas, dim=1).tolist()
    labels = sample_set['label'].tolist()

    table_data = []
    headers = ["Index", "Text", "Label", "Predicted Label"]
    max_len = 60
    for i, (sample_text, label, pred_label) in enumerate(zip(sample_texts, labels, pred_labels)):
        if len(sample_text) > max_len:
            sample_text = sample_text[:max_len] + '...'
        table_data.append([i, sample_text, label, pred_label])
    table = tabulate(table_data, headers, tablefmt="grid")
    print(table)  # seems accurate


# the functions specifically works on trainer_classification are listed below
# todo: trainer functions are not documented.

def freeze_layers(model, until: int = -1):
    for param in list(model.parameters())[:until]:  # experimental result
        param.requires_grad = False


def unfreeze_layers(model):
    for param in model.parameters():
        param.requires_grad = True


def change_args_for_warmup(trainer):
    """Design can be changed later. Dynamic patching seems good for now
    :param trainer:
    """
    training_args.num_train_epochs = 2
    training_args.lr_scheduler_type = 'constant'
    training_args.load_best_model_at_end = False
    training_args.save_strategy = 'no'
    trainer.remove_callback(EarlyStoppingCallback)


def start_warmup(trainer):
    print(f'using warmup for {model_name}')
    print('=' * 30)
    print('warmup is beginning')
    print('=' * 30)
    training_args_copy = deepcopy(training_args)
    change_args_for_warmup(trainer)
    freeze_layers(trainer.model)
    trainer.train()

    # revert everything to original
    trainer.args = training_args_copy
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=10))
    unfreeze_layers(trainer.model)


def compute_metrics(eval_predict, metrics: list):
    # we will not use inputs, but it may return inputs, so use a temp list to capture them, see the source code:
    #  def __iter__(self):
    #         if self.inputs is not None:
    #             return iter((self.predictions, self.label_ids, self.inputs))
    #         else:
    #             return iter((self.predictions, self.label_ids))

    predictions, labels, *unused = eval_predict
    del unused
    predictions = np.argmax(predictions, axis=1)

    metric_dict = {}
    for (name, metric) in metrics:
        result: dict = metric.compute(references=labels,
                                      predictions=predictions) if 'f1' not in name else metric.compute(
            references=labels, predictions=predictions, average='macro')
        metric_dict[name] = next(iter(result.values()))  # get the first value of the dict

    return metric_dict
