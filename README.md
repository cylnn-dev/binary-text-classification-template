# Binary Text Classification using PyTorch and HuggingFace

Detailed explanations on: [cylnn-dev](https://cylnn-dev.github.io/)

This project was written months ago when I was doing my internship at a conversational AI company. They wanted to test their in-house purified data with well-known transformers. Since then, I have deleted all private credentials and changed the dataset to the IMDB Large Movie Review. Feel free to try another binary text classification datasets.

NOTE: Do not forget to run split_csv_script.py before beginning. It will create train.csv and test.csv.

Add /datasets directory to store the target dataset. Point to the dataset location using configs.py.

## File Navigation


There are two `.py` files to train and test the models.

- `manual_classification.py` --> Fetch the model, preprocess the dataset and convert it to HuggingFace Dataset format, split the dataset into validation, and train the model


- `trainer_classification.py` --> Same as above, but this approach uses Trainer of HuggingFace. Avoid this approach, as you cannot control the things inside it.


- `configs.py` --> global variables used in various scripts. User can change these variables for their own ease. Seeing all variables in one place makes adjustments very easy and effective.


- `utility_functions.py` --> These functions are designed to be reusable. They abstract many things and pack them into several functions.


- `split_csv_script.py` --> As I have converted my old codes, I needed to add this script to split the dataset into train and test datasets. The training dataset will be split later on the *_classification.py files.


- `intel_compressor.py` --> Intel Neural Compressor is used for quantization, pruning, and knowledge distillation for various frameworks. I just added these two while there was nothing to do on the job. They can be useful if you are using a 10-year-old computer like me or you want to deploy your model to edge.


- `intel_compressor_test.py` --> It simply tries the compressed model.

