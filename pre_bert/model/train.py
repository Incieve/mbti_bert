import torch
import torch.nn as nn
import numpy as np

def train_epoch(model, data_loader, loss_fn, optimizer, 
    device, scheduler, n_examples
):  
    # Start of model training
    model = model.train()

    losses = []
    correct_predictions = 0
    
    for d in data_loader:
        # Push variables to device (GPU, CPU)
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        # Get the predictions from the BERT model
        outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
        )

        # Pick the most probable predictions for loss generation
        _, preds = torch.max(outputs, dim=1)

        # Compute loss for current step
        loss = loss_fn(outputs, targets)

        # Compute num of correct predictions -> this will be needed for model statistics
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        # Perform backpropagation
        loss.backward()

        # Clip gradient
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Step of the optimizer
        optimizer.step()

        # Step of the scheduler
        scheduler.step()

        # Unplug gradients from the mmemory tree (https://discuss.pytorch.org/t/what-step-backward-and-zero-grad-do/33301)
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    # Start of model evaluation
    model = model.eval()

    losses = []
    correct_predictions = 0

    # There is no need to train the model during the evaluation
    # That's why is foor loop in with torch.no_grad() wrapper
    with torch.no_grad():
        for d in data_loader:
            
            # Almost the same procedure as in the train function
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)