import random
import pandas as pd

from statistics import mean

import torch
import torch.nn as nn

import pytorch_lightning as pl

from transformers import BertModel
from sklearn.metrics import classification_report

# from torchmetrics import Accuracy

from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, PrecisionRecallCurve

from sklearn.metrics import f1_score, accuracy_score

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

class MultiClassModel(pl.LightningModule):
    def __init__(self,
                 dropout,
                 n_out,
                 lr) -> None:
        super(MultiClassModel, self).__init__()

        torch.manual_seed(1)
        random.seed(1)

        ## inisialisasi bert
        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased')

        ## pre_classifier = agar weight tidak hilang ketika epoch selanjutnya. Agar weight dapat digunakan kembali
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(dropout)

        ## classifier untuk merubah menjadi label
        self.classifier = nn.Linear(768, n_out)
        self.lr = lr

        ## menghitung loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # self.accuracy = torchmetrics.Accuracy(task="multiclass")

        # self.accuracy = MulticlassAccuracy(task="multiclass", num_classes = self.num_classes)
    
    ## Model
    ## mengambil input dari bert, pre_classifier
    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_out = self.bert(input_ids = input_ids,
                             attention_mask = attention_mask,
                             token_type_ids = token_type_ids)

        hidden_state = bert_out[0]
        pooler = hidden_state[:, 0]

        ## pre classifier untuk mentransfer wight output ke epch selanjuntya
        pooler = self.pre_classifier(pooler)
        
        ## kontrol hasil pooler min -1 max 1
        pooler = torch.nn.Tanh()(pooler)

        pooler = self.dropout(pooler)
        ## classifier untuk memprojeksikan hasil pooler (768) ke jumlah label (23)
        output = self.classifier(pooler)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = batch

        out = self(input_ids = x_input_ids,
                   attention_mask = x_attention_mask,
                   token_type_ids = x_token_type_ids)
        
        loss = self.criterion(out, target = y.float())

        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()
        f1_s = f1_score(true, pred, average='macro')
        
        avg_pred = sum(pred)/len(pred)
        predict = avg_pred.cpu().detach().numpy()
        # acc = accuracy_score(pred, true)

        # self.accuracy(out, y)
        # report = classification_report(true, pred, output_dict = True, zero_division = 0)

        # self.log("accuracy", report["accuracy"], prog_bar = True)
        # self.log("accuracy", acc, prog_bar = True)
        self.log("f1_score", f1_s, prog_bar = True)
        self.log("loss", loss)

        return {"loss": loss, "predictions": out, "F1": f1_s, "labels": y, "avg_pred": predict}

    def validation_step(self, batch, batch_idx):
        ## Tidak transfer weight
        x_input_ids, x_token_type_ids, x_attention_mask, y = batch

        out = self(input_ids = x_input_ids,
                   attention_mask = x_attention_mask,
                   token_type_ids = x_token_type_ids)

        loss = self.criterion(out, target = y.float())

        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()

        f1_s = f1_score(true, pred, average='macro')
        
        avg_pred = sum(pred)/len(pred)
        predict = avg_pred.cpu().detach().numpy()
        # acc = accuracy_score(pred, true)

        # report = classification_report(true, pred, output_dict = True, zero_division = 0)
        # self.accuracy(out, y)

        # self.log("accuracy", report["accuracy"], prog_bar = True)
        self.log("f1_score", f1_s, prog_bar = True)
        # self.log("accuracy", acc, prog_bar = True)
        self.log("loss", loss)

        return {"val_loss": loss, "predictions": out, "F1": f1_s, "labels": y, "avg_pred": predict}
    
    def test_step(self, batch, batch_idx):
        ## Tidak transfer weight
        x_input_ids, x_token_type_ids, x_attention_mask, y = batch

        out = self(input_ids = x_input_ids,
                   attention_mask = x_attention_mask,
                   token_type_ids = x_token_type_ids)

        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()
        f1_s = f1_score(true, pred, average='macro')
        
        # acc = accuracy_score(pred, true)

        self.log("f1_score", f1_s, prog_bar = True)
        # self.log("accuracy", acc, prog_bar = True)

        return {"predictions": pred, "labels": true, "F1": f1_s}

    def predict_step(self, batch, batch_idx):
        ## Tidak ada transfer weight
        x_input_ids, x_token_type_ids, x_attention_mask, y = batch

        out = self(input_ids = x_input_ids,
                   attention_mask = x_attention_mask,
                   token_type_ids = x_token_type_ids)

        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()
        
        return {"predictions": pred, "labels": true}