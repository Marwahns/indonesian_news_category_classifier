import random
import pandas as pd

from statistics import mean

import torch
import torch.nn as nn

import pytorch_lightning as pl

from transformers import BertModel

from sklearn.metrics import classification_report, f1_score, accuracy_score

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

        self.train_score = {
            # 'labels': [],
            'loss': [],
            'f1_micro': [],
            'f1_macro': [],
            'accuracy': []
        }

        self.validation_score = {
            # 'labels': [],
            'loss': [],
            'f1_micro': [],
            'f1_macro': [],
            'accuracy': []
        }

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
        f1_s_micro = f1_score(true, pred, average='micro')
        f1_s_macro = f1_score(true, pred, average='macro')

        acc = accuracy_score(pred, true)

        # report = classification_report(true, pred, output_dict = True, zero_division = 0)

        # self.log('accuracy', report['accuracy'], prog_bar = True)
        self.log('accuracy', acc, prog_bar = True)
        self.log('f1_score_micro', f1_s_micro, prog_bar = True)
        self.log('f1_score_macro', f1_s_macro, prog_bar = True)
        self.log('loss', loss)

        # return {'loss': loss, 'predictions': out, 'f1_micro': f1_s_micro, 'f1_macro': f1_s_macro, 'accuracy': report['accuracy'], 'labels': y, 'avg_pred': predict}
        return {'loss': loss, 'predictions': out, 'f1_micro': f1_s_micro, 'f1_macro': f1_s_macro, 'accuracy': acc, 'labels': y}
    
    ## jumlah return di validation_step harus sama dengan jumlah return di training_step
    def validation_step(self, batch, batch_idx):
        ## Tidak transfer weight
        x_input_ids, x_token_type_ids, x_attention_mask, y = batch

        out = self(input_ids = x_input_ids,
                   attention_mask = x_attention_mask,
                   token_type_ids = x_token_type_ids)

        loss = self.criterion(out, target = y.float())

        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu() 

        f1_s_micro = f1_score(true, pred, average='micro')
        f1_s_macro = f1_score(true, pred, average='macro')
        acc = accuracy_score(pred, true)

        # report = classification_report(true, pred, output_dict = True, zero_division = 0)

        self.log('f1_score_micro', f1_s_micro, prog_bar = True)
        self.log('f1_score_macro', f1_s_macro, prog_bar = True)
        # self.log('accuracy', report['accuracy'], prog_bar = True)
        self.log('accuracy', acc, prog_bar = True)
        self.log('loss', loss)

        # return {'val_loss': loss, 'predictions': out, 'f1_micro': f1_s_micro, 'f1_macro': f1_s_macro, 'accuracy': report['accuracy'], 'avg_pred': predict}
        return {'val_loss': loss, 'predictions': out, 'f1_micro': f1_s_micro, 'f1_macro': f1_s_macro, 'accuracy': acc, 'labels': y}
    
    def test_step(self, batch, batch_idx):
        ## Tidak transfer weight
        x_input_ids, x_token_type_ids, x_attention_mask, y = batch

        out = self(input_ids = x_input_ids,
                   attention_mask = x_attention_mask,
                   token_type_ids = x_token_type_ids)

        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()

        f1_s_micro = f1_score(true, pred, average='micro')
        f1_s_macro = f1_score(true, pred, average='macro')
        acc = accuracy_score(pred, true)

        self.log('f1_score_micro', f1_s_micro, prog_bar = True)
        self.log('f1_score_macro', f1_s_macro, prog_bar = True)
        self.log('accuracy', acc, prog_bar = True)
        
        return {'predictions': pred, 'labels': true, 'f1_micro': f1_s_micro, 'f1_macro': f1_s_macro, 'accuracy': acc}

    def predict_step(self, batch, batch_idx):
        ## Tidak ada transfer weight
        x_input_ids, x_token_type_ids, x_attention_mask, y = batch

        out = self(input_ids = x_input_ids,
                   attention_mask = x_attention_mask,
                   token_type_ids = x_token_type_ids)

        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()
        
        return {'predictions': pred, 'labels': true}
    
    def create_figure(self, data, fig_dir, y_label):
        ## c_fig untuk nama figure (graph)
        ## c_ax untuk koordinat pada graph
        c_fig, c_ax = plt.subplots()
        c_ax.set_xlabel('epoch')
        c_ax.set_ylabel(y_label)
        c_ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # c_ax.plot(data, marker='o', mfc='green', mec='yellow', ms='7')
        c_ax.plot(data, marker='o', ms='7')

        for x_epoch, y_sc in enumerate(data):
            y_sc_lbl = '{:.2f}'.format(y_sc)

            c_ax.annotate(y_sc_lbl,
                         (x_epoch, y_sc),
                          textcoords='offset points',
                          xytext=(0,9),
                          ha='center'
                        #   arrowprops=dict(arrowstyle='->',
                        #   color='black')
                          )
        
        c_fig.savefig(fig_dir)

    def training_epoch_end(self, outputs):
        scores = {
            # 'labels': [],
            'loss': [],
            'f1_micro': [],
            'f1_macro': [],
            'accuracy': []
        }

        for output in outputs:
            # scores['labels'].append(output['labels'])
            scores['f1_micro'].append(output['f1_micro'])
            scores['f1_macro'].append(output['f1_macro'])
            ## karena loss di kerjakan oleh gpu, maka harus di ubah ke cpu terlebih dahulu. agar loss dapat ditampilkan
            scores['loss'].append(output['loss'].detach().cpu().item())
            scores['accuracy'].append(output['accuracy'])
        
        # self.train_score['labels'].append(scores['labels'])
        self.train_score['f1_micro'].append(mean(scores['f1_micro']))
        self.train_score['f1_macro'].append(mean(scores['f1_macro']))
        self.train_score['loss'].append(mean(scores['loss']))
        self.train_score['accuracy'].append(mean(scores['accuracy']))

        df_scores = pd.DataFrame.from_dict(self.train_score)
        # df_scores.to_csv('.graph/'+'training_scores.csv')
        df_scores.to_csv('training_scores.csv')

        # self.create_figure(self.train_score['f1_micro'], './graph/image' + 'training_f1_score_micro.png', 'f1-score micro')
        self.create_figure(self.train_score['f1_micro'], 'training_f1_score_micro.png', 'f1-score micro')
        self.create_figure(self.train_score['f1_macro'], 'training_f1_score_macro.png', 'f1-score macro')
        self.create_figure(self.train_score['loss'], 'training_loss.png', 'loss')
        self.create_figure(self.train_score['accuracy'], 'training_accuracy.png', 'accuracy')
    
    def validation_epoch_end(self, outputs):
        scores = {
            # 'labels': [],
            'loss': [],
            'f1_micro': [],
            'f1_macro': [],
            'accuracy': []
        }

        for output in outputs:
            # scores['labels'].append(output['labels'])
            scores['f1_micro'].append(output['f1_micro'])
            scores['f1_macro'].append(output['f1_macro'])
            scores['loss'].append(output['val_loss'].detach().cpu().item())
            scores['accuracy'].append(output['accuracy'])
        
        # self.validation_score['labels'].append(scores['labels'])
        self.validation_score['f1_micro'].append(mean(scores['f1_micro']))
        self.validation_score['f1_macro'].append(mean(scores['f1_macro']))
        self.validation_score['loss'].append(mean(scores['loss']))
        self.validation_score['accuracy'].append(mean(scores['accuracy']))

        df_scores = pd.DataFrame.from_dict(self.validation_score)
        # df_scores.to_csv('.graph/'+'validation_scores.csv')
        df_scores.to_csv('validation_scores.csv')

        # self.create_figure(self.train_score['f1_micro'], './graph/image' + 'validation_f1_score_micro.png', 'f1-score micro')
        self.create_figure(self.validation_score['f1_micro'], 'validation_f1_score_micro.png', 'f1-score micro')
        self.create_figure(self.validation_score['f1_macro'], 'validation_f1_score_macro.png', 'f1-score macro')
        self.create_figure(self.validation_score['loss'], 'validation_loss.png', 'loss')
        self.create_figure(self.validation_score['accuracy'], 'validation_accuracy.png', 'accuracy')
    
    def test_epoch_end(self, outputs):
        scores = {
            'f1_micro': [],
            'f1_macro': [],
            'accuracy': []
        }

        for output in outputs:
            scores['f1_micro'].append(output['f1_micro'])
            scores['f1_macro'].append(output['f1_macro'])
            scores['accuracy'].append(output['accuracy'])
        
        print('F1-Score Micro = ', mean(scores['f1_micro']), 'F1-Score Macro = ', mean(scores['f1_macro']), ' Accuracy = ', mean(scores['accuracy']))