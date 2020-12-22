import os
from textwrap import wrap
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

from utils import clean_text
from loader import create_data_loader
from model import PersonalityClassifier
from train import train_epoch, eval_model


# Random seed - for the same model initialization in each experiment
RANDOM_SEED = 42

# Batch size - how many data samples are in one batch
BATCH_SIZE = 32

# Epochs - For how many epochs is model trained (epoch - one iteration through whole dataset)
EPOCHS = 10

# Length of the sample in batch
MAX_LEN = 160

# Specification of pretrained model (https://huggingface.co/transformers/pretrained_models.html)
PRETRAINED_MODEL = 'bert-base-uncased'

# Dict with mbti types - used for label creation
LABEL_ID = {"ISTJ": 0, "ISTP": 1, "ISFJ": 2, "ISFP": 3,
            "INFJ": 4, "INFP": 5, "INTJ": 6, "INTP": 7,
            "ESTP": 8, "ESTJ": 9, "ESFP": 10, "ESFJ": 11,
            "ENFP": 12, "ENFJ": 13, "ENTP": 14, "ENTJ": 15}

# This is used to hide warnings in tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set seed for numpy and pytorch
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# specify GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create pandas DataFrame from .csv file and apply LABEL_ID dict on types
df = pd.read_csv("pre_bert/data/mbti_1.csv").replace({"type": LABEL_ID})

# Clean text in posts column in database i.e. lower
df["posts"] = df["posts"].apply(clean_text)

# Creates tokens from plain text
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)

# Split database to train, validation and test datasets
df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

# Create batches which serves as inputs to the model.
# Backpropagation is realized after each batch
train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

data = next(iter(train_data_loader))

# Calculate how many classes are in the dataset
n_classes = df["type"].nunique()

# Initialize model with specific pretrained BERT and number of classes
# Push model to specified device (GPU, CPU)
model = PersonalityClassifier(n_classes=n_classes, bert_type=PRETRAINED_MODEL)
model = model.to(device)

# Push input and mask tensors to specified device (GPU, CPU)
input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)

# Initialize optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

# Scheduler performs linear decay of learning rate
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Defines loss and push to device (GPU, CPU)
loss_fn = nn.CrossEntropyLoss().to(device)

# Main loop of the script
if __name__ == "__main__":
    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(df_val)
        )

        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state_dr01.pt')
            best_accuracy = val_acc