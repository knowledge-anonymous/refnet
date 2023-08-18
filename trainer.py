# coding=utf-8

"""  """
from typing import Callable

import torch

class Trainer:
    def __init__(self, model, optimizer, loss_fn, device, label_fn: Callable, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
        self.label_fn = label_fn

    def fit(self, train_dataloader, val_dataloader, epochs):
        self.model.train()  # Set the model to training mode
        self.model = self.model.to(self.device)
        for epoch in range(epochs):
            total_loss = 0
            # Training Phase
            for batch in train_dataloader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()  # Reset the gradients
                outputs = self.model(batch)  # Forward pass
                loss = self.loss_fn(outputs, self.label_fn(batch))
                loss.backward()  # Backward pass
                self.optimizer.step()  # Update the weights
                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataloader)
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss}')

            # Validation Phase
            avg_val_loss = self.evaluate(val_dataloader)
            print(f'Validation Loss: {avg_val_loss}')

            if self.scheduler is not None:
                self.scheduler.step(avg_val_loss)  # Update the learning rate

    def evaluate(self, dataloader):
        self.model.eval()  # Set the model to evaluation mode
        self.model = self.model.to(self.device)
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)

                outputs = self.model(batch)

                loss = self.loss_fn(outputs, self.label_fn(batch))
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def test(self, dataloader):
        avg_test_loss, test_accuracy = self.evaluate(dataloader)
        print(f'Test Loss: {avg_test_loss}, Test Accuracy: {test_accuracy * 100}%')
