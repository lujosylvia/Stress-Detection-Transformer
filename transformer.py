import sys
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import one_hot
from torchvision.ops import sigmoid_focal_loss
from sklearn.metrics import accuracy_score, f1_score

from research_paper.final.dataset_builders import split_dataset

Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#
# Transformer Encoder Model
# Trained for Binary Classification of stress using WESAD dataset
#
class TransformerStressPredictor(nn.Module):
    def __init__(self, input_dims, model_dim=16, nhead=2, n_encoder_layers=4, n_classes=2, dropout=0.25, batch_size=2000):
        super().__init__()
        self.input_dims = input_dims
        self.model_dim = model_dim
        self.batch_size = batch_size
        self.totloss = 0
        self.all_losses = []

        self.embeddings = nn.ModuleList([
            nn.Linear(1, model_dim) for _ in range(input_dims)
        ]).to(Device)
        self.dropout = nn.Dropout(dropout)
        self.pos_encoding = nn.Parameter(torch.rand(1, self.batch_size, model_dim)).to(Device) # embeddings will all be concatenated later
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=nhead,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=0,
            batch_first=True
        ).to(Device)
        self.classifier = nn.Linear(model_dim, n_classes).to(Device)

    def forward(self, _x):
        _x = _x.unsqueeze(0).to(Device)

        # Apply separate embeddings to each signal
        embeddings = []
        for i in range(self.input_dims):
            signal = _x[:, :, i].unsqueeze(-1)
            embedding = self.embeddings[i](signal)
            embeddings.append(embedding)

        # Combine all signal embeddings by concatenating
        combined_embedding = torch.cat(embeddings, dim=0).to(Device)

        # Add positional encoding
        combined_embedding += self.pos_encoding.to(Device)

        combined_embedding = self.dropout(combined_embedding)

        transformer_output = self.transformer.encoder(combined_embedding)
        output = transformer_output.mean(dim=0) 

        return self.classifier(output)

    def predict(self, _x):
      with torch.no_grad():
        logits = self.forward(_x)
        return torch.argmax(logits, dim=1)
      

#
# Method used to run training epochs on the Transformer Encoder model
# Defaults to running 10 epochs
# Defaults to using a learning rate of 0.001
# Defaults to using a weight decay of 0.00001
# Uses AdamW as an optiizer
# Uses ReduceLROnPlateau as a learning rate scheduler
#
def train(_model, _dataloader, epochs=10, lr=1e-3, weight_decay=1e-5, info=True):
    optimizer = torch.optim.AdamW(_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, mode='min')
    criterion = sigmoid_focal_loss

    _model.train()

    training_loss = []

    for e in range(epochs):
        totloss = 0
        for x, y in _dataloader:
            optimizer.zero_grad()
            x, y = x.to(Device), y.to(Device)
            out = _model.forward(x)

            o_y = y.clone().detach().type(torch.LongTensor)

            targets = one_hot(o_y, num_classes=2)
            targets = targets.float().to('cpu')

            probabilities = out.type(torch.FloatTensor).to('cpu')

            loss = criterion(probabilities, targets, alpha=0.7, reduction='mean')
            loss.backward()

            optimizer.step()
            totloss += loss.item()

        training_loss.append(totloss)
        scheduler.step(totloss)

        if info:
            sys.stderr.write(f"\r{e+1:3d}/{epochs:3d} | Loss: {totloss:3.3f}")
            sys.stderr.flush()
    
    _model.totloss = np.mean(training_loss)
    _model.all_losses = training_loss

#
# evaluates the performance of the Transformer Encoder model after training over a number of epochs.
# Calculates the model's predictions and returns them with the actual values.
#
#
def evaluate(_model, _testloader):
    _model.eval()

    y_preds = []
    y_actual = []
    with torch.no_grad():
      for _x, _y in _testloader:

          _x, _y = _x.to(Device), _y.to(Device)

          y_pred = _model.predict(_x)

          y_preds.extend(y_pred.to('cpu').numpy())
          y_actual.extend(_y.to('cpu').numpy())

    return np.array(y_preds), np.array(y_actual)

#
# cross validation using time series
# defaults to a K split of 5
# the data split is done by randomly grouping subject data into different data subsets
#
def time_series_split_train_test_validation(_clf, _data_dict: dict, _signals: list[str] = None, epochs: int = 25, splits: int = 5, _batch_size=2000, _train_size = 0.75):
    accs, f1s, losses = [], [], []
    datasets = split_dataset(list(_data_dict.keys()), splits)

    for dataset in datasets:

        _combined_df = pd.concat([_data_dict[entry] for entry in dataset], axis=0, ignore_index=True)
        _X_columns = _signals if _signals != None else [col for col in _combined_df.columns if col != 'label']

        X = _combined_df[_X_columns].values
        y = _combined_df['label'].values.ravel()

        split_index = int(len(X) * _train_size)

        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        X_tr = torch.tensor(X_train, dtype=torch.float32)
        y_tr = torch.tensor(y_train, dtype=torch.float32)

        X_ts = torch.tensor(X_test, dtype=torch.float32)
        y_ts = torch.tensor(y_test, dtype=torch.float32)

        ds_train = TensorDataset(X_tr, y_tr)
        dl_train = DataLoader(ds_train, batch_size=_batch_size, shuffle=False, num_workers=4, drop_last=True)

        ds_test = TensorDataset(X_ts, y_ts)
        dl_test =  DataLoader(ds_test, batch_size=_batch_size, shuffle=False, num_workers=4, drop_last=True)

        train(_clf, dl_train, epochs=epochs, info=True)

        y_pred, y_actual = evaluate(_clf, dl_test)

        acc = accuracy_score(y_actual, y_pred)
        f1 = f1_score(y_actual, y_pred, average="weighted")

        accs.append(acc)
        f1s.append(f1)
        losses.append(_clf.totloss)

        print(f'Acc= {acc:.2f}')
        print(f'F1= {f1:.2f}')
        print(f'Loss= {_clf.totloss:.2f}')

    return np.mean(accs), np.mean(f1s), np.mean(losses), _clf.all_losses