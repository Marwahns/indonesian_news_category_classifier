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
            'loss': [],
            'f1_micro': [],
            'f1_macro': [],
            'accuracy': []
        }

        self.train_label = {
            'daerah': [],
            'ekbis': [],
            'entertainment': [],
            'foto': [],
            'global': [],
            'hankam': [],
            'history': [],
            'hukum': [],
            'kesehatan': [],
            'khazanah': [],
            'lifestyle': [],
            'metro': [],
            'militer': [],
            'nasional': [],
            'otomotif': [],
            'peristiwa': [],
            'politik': [],
            'property': [],
            'seleb': [],
            'sosmed': [],
            'sport': [],
            'techno': []
        }

        self.train_labels_count = {
            'daerah': [],
            'ekbis': [],
            'entertainment': [],
            'foto': [],
            'global': [],
            'hankam': [],
            'history': [],
            'hukum': [],
            'kesehatan': [],
            'khazanah': [],
            'lifestyle': [],
            'metro': [],
            'militer': [],
            'nasional': [],
            'otomotif': [],
            'peristiwa': [],
            'politik': [],
            'property': [],
            'seleb': [],
            'sosmed': [],
            'sport': [],
            'techno': []
        }

        self.validation_score = {
            'loss': [],
            'f1_micro': [],
            'f1_macro': [],
            'accuracy': []
        }

        self.validation_label = {
            'daerah': [],
            'ekbis': [],
            'entertainment': [],
            'foto': [],
            'global': [],
            'hankam': [],
            'history': [],
            'hukum': [],
            'kesehatan': [],
            'khazanah': [],
            'lifestyle': [],
            'metro': [],
            'militer': [],
            'nasional': [],
            'otomotif': [],
            'peristiwa': [],
            'politik': [],
            'property': [],
            'seleb': [],
            'sosmed': [],
            'sport': [],
            'techno': []
        }

        self.validation_labels_count = {
            'daerah': [],
            'ekbis': [],
            'entertainment': [],
            'foto': [],
            'global': [],
            'hankam': [],
            'history': [],
            'hukum': [],
            'kesehatan': [],
            'khazanah': [],
            'lifestyle': [],
            'metro': [],
            'militer': [],
            'nasional': [],
            'otomotif': [],
            'peristiwa': [],
            'politik': [],
            'property': [],
            'seleb': [],
            'sosmed': [],
            'sport': [],
            'techno': []
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
            'loss': [],
            'f1_micro': [],
            'f1_macro': [],
            'accuracy': []
        }

        labels = {
            'daerah': [],
            'ekbis': [],
            'entertainment': [],
            'foto': [],
            'global': [],
            'hankam': [],
            'history': [],
            'hukum': [],
            'kesehatan': [],
            'khazanah': [],
            'lifestyle': [],
            'metro': [],
            'militer': [],
            'nasional': [],
            'otomotif': [],
            'peristiwa': [],
            'politik': [],
            'property': [],
            'seleb': [],
            'sosmed': [],
            'sport': [],
            'techno': []
        }

        labels_count = {
            'daerah': [],
            'ekbis': [],
            'entertainment': [],
            'foto': [],
            'global': [],
            'hankam': [],
            'history': [],
            'hukum': [],
            'kesehatan': [],
            'khazanah': [],
            'lifestyle': [],
            'metro': [],
            'militer': [],
            'nasional': [],
            'otomotif': [],
            'peristiwa': [],
            'politik': [],
            'property': [],
            'seleb': [],
            'sosmed': [],
            'sport': [],
            'techno': []
        }

        for output in outputs:
            ## append = menambahkan item dari belakang
            ## karena loss di kerjakan oleh gpu, maka harus di ubah ke cpu terlebih dahulu. agar loss dapat ditampilkan

            ## labels
            labels['daerah'].append(output['labels'][0][0].detach().cpu().item())
            labels['ekbis'].append(output['labels'][0][1].detach().cpu().item())
            labels['entertainment'].append(output['labels'][0][2].detach().cpu().item())
            labels['foto'].append(output['labels'][0][3].detach().cpu().item())
            labels['global'].append(output['labels'][0][4].detach().cpu().item())
            labels['hankam'].append(output['labels'][0][5].detach().cpu().item())
            labels['history'].append(output['labels'][0][6].detach().cpu().item())
            labels['hukum'].append(output['labels'][0][7].detach().cpu().item())
            labels['kesehatan'].append(output['labels'][0][8].detach().cpu().item())
            labels['khazanah'].append(output['labels'][0][9].detach().cpu().item())
            labels['lifestyle'].append(output['labels'][0][10].detach().cpu().item())
            labels['metro'].append(output['labels'][0][11].detach().cpu().item())
            labels['militer'].append(output['labels'][0][12].detach().cpu().item())
            labels['nasional'].append(output['labels'][0][13].detach().cpu().item())
            labels['otomotif'].append(output['labels'][0][14].detach().cpu().item())
            labels['peristiwa'].append(output['labels'][0][15].detach().cpu().item())
            labels['politik'].append(output['labels'][0][16].detach().cpu().item())
            labels['property'].append(output['labels'][0][17].detach().cpu().item())
            labels['seleb'].append(output['labels'][0][18].detach().cpu().item())
            labels['sosmed'].append(output['labels'][0][19].detach().cpu().item())
            labels['sport'].append(output['labels'][0][20].detach().cpu().item())
            labels['techno'].append(output['labels'][0][21].detach().cpu().item())

            ## scores
            scores['f1_micro'].append(output['f1_micro'])
            scores['f1_macro'].append(output['f1_macro'])
            scores['loss'].append(output['loss'].detach().cpu().item())
            scores['accuracy'].append(output['accuracy'])
        
        ## labels
        self.train_label['daerah'].append(labels['daerah'])
        self.train_label['ekbis'].append(labels['ekbis'])
        self.train_label['entertainment'].append(labels['entertainment'])
        self.train_label['foto'].append(labels['foto'])
        self.train_label['global'].append(labels['global'])
        self.train_label['hankam'].append(labels['hankam'])
        self.train_label['history'].append(labels['history'])
        self.train_label['hukum'].append(labels['hukum'])
        self.train_label['kesehatan'].append(labels['kesehatan'])
        self.train_label['khazanah'].append(labels['khazanah'])
        self.train_label['lifestyle'].append(labels['lifestyle'])
        self.train_label['metro'].append(labels['metro'])
        self.train_label['militer'].append(labels['militer'])
        self.train_label['nasional'].append(labels['nasional'])
        self.train_label['otomotif'].append(labels['otomotif'])
        self.train_label['peristiwa'].append(labels['peristiwa'])
        self.train_label['politik'].append(labels['politik'])
        self.train_label['property'].append(labels['property'])
        self.train_label['seleb'].append(labels['seleb'])
        self.train_label['sosmed'].append(labels['sosmed'])
        self.train_label['sport'].append(labels['sport'])
        self.train_label['techno'].append(labels['techno'])

        ## scores
        self.train_score['f1_micro'].append(mean(scores['f1_micro']))
        self.train_score['f1_macro'].append(mean(scores['f1_macro']))
        self.train_score['loss'].append(mean(scores['loss']))
        self.train_score['accuracy'].append(mean(scores['accuracy']))

        ## Label count
        labels_count['daerah'] = labels['daerah'].count(1)
        labels_count['ekbis'] = labels['ekbis'].count(1)
        labels_count['entertainment'] = labels['entertainment'].count(1)
        labels_count['foto'] = labels['foto'].count(1)
        labels_count['global'] = labels['global'].count(1)
        labels_count['hankam'] = labels['hankam'].count(1)
        labels_count['history'] = labels['history'].count(1)
        labels_count['hukum'] = labels['hukum'].count(1)
        labels_count['kesehatan'] = labels['kesehatan'].count(1)
        labels_count['khazanah'] = labels['khazanah'].count(1)
        labels_count['lifestyle'] = labels['lifestyle'].count(1)
        labels_count['metro'] = labels['metro'].count(1)
        labels_count['militer'] = labels['militer'].count(1)
        labels_count['nasional'] = labels['nasional'].count(1)
        labels_count['otomotif'] = labels['otomotif'].count(1)
        labels_count['peristiwa'] = labels['peristiwa'].count(1)
        labels_count['politik'] = labels['politik'].count(1)
        labels_count['property'] = labels['property'].count(1)
        labels_count['seleb'] = labels['seleb'].count(1)
        labels_count['sosmed'] = labels['sosmed'].count(1)
        labels_count['sport'] = labels['sport'].count(1)
        labels_count['techno'] = labels['techno'].count(1)

        ## Labels count
        self.train_labels_count['daerah'].append(labels_count['daerah'])
        self.train_labels_count['ekbis'].append(labels_count['ekbis'])
        self.train_labels_count['entertainment'].append(labels_count['entertainment'])
        self.train_labels_count['foto'].append(labels_count['foto'])
        self.train_labels_count['global'].append(labels_count['global'])
        self.train_labels_count['hankam'].append(labels_count['hankam'])
        self.train_labels_count['history'].append(labels_count['history'])
        self.train_labels_count['hukum'].append(labels_count['hukum'])
        self.train_labels_count['kesehatan'].append(labels_count['kesehatan'])
        self.train_labels_count['khazanah'].append(labels_count['khazanah'])
        self.train_labels_count['lifestyle'].append(labels_count['lifestyle'])
        self.train_labels_count['metro'].append(labels_count['metro'])
        self.train_labels_count['militer'].append(labels_count['militer'])
        self.train_labels_count['nasional'].append(labels_count['nasional'])
        self.train_labels_count['otomotif'].append(labels_count['otomotif'])
        self.train_labels_count['peristiwa'].append(labels_count['peristiwa'])
        self.train_labels_count['politik'].append(labels_count['politik'])
        self.train_labels_count['property'].append(labels_count['property'])
        self.train_labels_count['seleb'].append(labels_count['seleb'])
        self.train_labels_count['sosmed'].append(labels_count['sosmed'])
        self.train_labels_count['sport'].append(labels_count['sport'])
        self.train_labels_count['techno'].append(labels_count['techno'])

        df_scores = pd.DataFrame.from_dict(self.train_score)
        df_labels = pd.DataFrame.from_dict(self.train_label)
        df_labels_count = pd.DataFrame.from_dict(self.train_labels_count)

        df_scores.to_csv('training_scores.csv')
        df_labels.to_csv('training_labels.csv')
        df_labels_count.to_csv('training_labels_count.csv')

        self.create_figure(self.train_score['f1_micro'], 'training_f1_score_micro.png', 'f1-score micro')
        self.create_figure(self.train_score['f1_macro'], 'training_f1_score_macro.png', 'f1-score macro')
        self.create_figure(self.train_score['loss'], 'training_loss.png', 'loss')
        self.create_figure(self.train_score['accuracy'], 'training_accuracy.png', 'accuracy')

        print('F1-Score Micro = ', "{:.4f}".format(mean(scores['f1_micro'])), '| F1-Score Macro = ', "{:.4f}".format(mean(scores['f1_macro'])), '| Loss = ', "{:.4f}".format(mean(scores['loss'])), f'| Accuracy = {mean(scores["accuracy"])*100:.2f}%')

        # print('Politik = ', str(labels['politik'].count(1)))

        ## print "{:.2f}".format(56.455323)
        ## print(f'accuracy: {a*100:.2f}%')
    
    def validation_epoch_end(self, outputs):
        scores = {
            'loss': [],
            'f1_micro': [],
            'f1_macro': [],
            'accuracy': []
        }

        labels = {
            'daerah': [],
            'ekbis': [],
            'entertainment': [],
            'foto': [],
            'global': [],
            'hankam': [],
            'history': [],
            'hukum': [],
            'kesehatan': [],
            'khazanah': [],
            'lifestyle': [],
            'metro': [],
            'militer': [],
            'nasional': [],
            'otomotif': [],
            'peristiwa': [],
            'politik': [],
            'property': [],
            'seleb': [],
            'sosmed': [],
            'sport': [],
            'techno': []
        }

        labels_count = {
            'daerah': [],
            'ekbis': [],
            'entertainment': [],
            'foto': [],
            'global': [],
            'hankam': [],
            'history': [],
            'hukum': [],
            'kesehatan': [],
            'khazanah': [],
            'lifestyle': [],
            'metro': [],
            'militer': [],
            'nasional': [],
            'otomotif': [],
            'peristiwa': [],
            'politik': [],
            'property': [],
            'seleb': [],
            'sosmed': [],
            'sport': [],
            'techno': []
        }

        for output in outputs:
            ## labels
            labels['daerah'].append(output['labels'][0][0].detach().cpu().item())
            labels['ekbis'].append(output['labels'][0][1].detach().cpu().item())
            labels['entertainment'].append(output['labels'][0][2].detach().cpu().item())
            labels['foto'].append(output['labels'][0][3].detach().cpu().item())
            labels['global'].append(output['labels'][0][4].detach().cpu().item())
            labels['hankam'].append(output['labels'][0][5].detach().cpu().item())
            labels['history'].append(output['labels'][0][6].detach().cpu().item())
            labels['hukum'].append(output['labels'][0][7].detach().cpu().item())
            labels['kesehatan'].append(output['labels'][0][8].detach().cpu().item())
            labels['khazanah'].append(output['labels'][0][9].detach().cpu().item())
            labels['lifestyle'].append(output['labels'][0][10].detach().cpu().item())
            labels['metro'].append(output['labels'][0][11].detach().cpu().item())
            labels['militer'].append(output['labels'][0][12].detach().cpu().item())
            labels['nasional'].append(output['labels'][0][13].detach().cpu().item())
            labels['otomotif'].append(output['labels'][0][14].detach().cpu().item())
            labels['peristiwa'].append(output['labels'][0][15].detach().cpu().item())
            labels['politik'].append(output['labels'][0][16].detach().cpu().item())
            labels['property'].append(output['labels'][0][17].detach().cpu().item())
            labels['seleb'].append(output['labels'][0][18].detach().cpu().item())
            labels['sosmed'].append(output['labels'][0][19].detach().cpu().item())
            labels['sport'].append(output['labels'][0][20].detach().cpu().item())
            labels['techno'].append(output['labels'][0][21].detach().cpu().item())

            ## scores
            scores['f1_micro'].append(output['f1_micro'])
            scores['f1_macro'].append(output['f1_macro'])
            scores['loss'].append(output['val_loss'].detach().cpu().item())
            scores['accuracy'].append(output['accuracy'])
        
        ## labels
        self.validation_label['daerah'].append(labels['daerah'])
        self.validation_label['ekbis'].append(labels['ekbis'])
        self.validation_label['entertainment'].append(labels['entertainment'])
        self.validation_label['foto'].append(labels['foto'])
        self.validation_label['global'].append(labels['global'])
        self.validation_label['hankam'].append(labels['hankam'])
        self.validation_label['history'].append(labels['history'])
        self.validation_label['hukum'].append(labels['hukum'])
        self.validation_label['kesehatan'].append(labels['kesehatan'])
        self.validation_label['khazanah'].append(labels['khazanah'])
        self.validation_label['lifestyle'].append(labels['lifestyle'])
        self.validation_label['metro'].append(labels['metro'])
        self.validation_label['militer'].append(labels['militer'])
        self.validation_label['nasional'].append(labels['nasional'])
        self.validation_label['otomotif'].append(labels['otomotif'])
        self.validation_label['peristiwa'].append(labels['peristiwa'])
        self.validation_label['politik'].append(labels['politik'])
        self.validation_label['property'].append(labels['property'])
        self.validation_label['seleb'].append(labels['seleb'])
        self.validation_label['sosmed'].append(labels['sosmed'])
        self.validation_label['sport'].append(labels['sport'])
        self.validation_label['techno'].append(labels['techno'])

        ## scores
        self.validation_score['f1_micro'].append(mean(scores['f1_micro']))
        self.validation_score['f1_macro'].append(mean(scores['f1_macro']))
        self.validation_score['loss'].append(mean(scores['loss']))
        self.validation_score['accuracy'].append(mean(scores['accuracy']))

        ## Label count
        labels_count['daerah'] = labels['daerah'].count(1)
        labels_count['ekbis'] = labels['ekbis'].count(1)
        labels_count['entertainment'] = labels['entertainment'].count(1)
        labels_count['foto'] = labels['foto'].count(1)
        labels_count['global'] = labels['global'].count(1)
        labels_count['hankam'] = labels['hankam'].count(1)
        labels_count['history'] = labels['history'].count(1)
        labels_count['hukum'] = labels['hukum'].count(1)
        labels_count['kesehatan'] = labels['kesehatan'].count(1)
        labels_count['khazanah'] = labels['khazanah'].count(1)
        labels_count['lifestyle'] = labels['lifestyle'].count(1)
        labels_count['metro'] = labels['metro'].count(1)
        labels_count['militer'] = labels['militer'].count(1)
        labels_count['nasional'] = labels['nasional'].count(1)
        labels_count['otomotif'] = labels['otomotif'].count(1)
        labels_count['peristiwa'] = labels['peristiwa'].count(1)
        labels_count['politik'] = labels['politik'].count(1)
        labels_count['property'] = labels['property'].count(1)
        labels_count['seleb'] = labels['seleb'].count(1)
        labels_count['sosmed'] = labels['sosmed'].count(1)
        labels_count['sport'] = labels['sport'].count(1)
        labels_count['techno'] = labels['techno'].count(1)

        ## Labels count
        self.validation_labels_count['daerah'].append(labels_count['daerah'])
        self.validation_labels_count['ekbis'].append(labels_count['ekbis'])
        self.validation_labels_count['entertainment'].append(labels_count['entertainment'])
        self.validation_labels_count['foto'].append(labels_count['foto'])
        self.validation_labels_count['global'].append(labels_count['global'])
        self.validation_labels_count['hankam'].append(labels_count['hankam'])
        self.validation_labels_count['history'].append(labels_count['history'])
        self.validation_labels_count['hukum'].append(labels_count['hukum'])
        self.validation_labels_count['kesehatan'].append(labels_count['kesehatan'])
        self.validation_labels_count['khazanah'].append(labels_count['khazanah'])
        self.validation_labels_count['lifestyle'].append(labels_count['lifestyle'])
        self.validation_labels_count['metro'].append(labels_count['metro'])
        self.validation_labels_count['militer'].append(labels_count['militer'])
        self.validation_labels_count['nasional'].append(labels_count['nasional'])
        self.validation_labels_count['otomotif'].append(labels_count['otomotif'])
        self.validation_labels_count['peristiwa'].append(labels_count['peristiwa'])
        self.validation_labels_count['politik'].append(labels_count['politik'])
        self.validation_labels_count['property'].append(labels_count['property'])
        self.validation_labels_count['seleb'].append(labels_count['seleb'])
        self.validation_labels_count['sosmed'].append(labels_count['sosmed'])
        self.validation_labels_count['sport'].append(labels_count['sport'])
        self.validation_labels_count['techno'].append(labels_count['techno'])

        df_scores = pd.DataFrame.from_dict(self.validation_score)
        df_labels = pd.DataFrame.from_dict(self.validation_label)
        df_labels_count = pd.DataFrame.from_dict(self.validation_labels_count)

        df_scores.to_csv('validation_scores.csv')
        df_labels.to_csv('validation_labels.csv')
        df_labels_count.to_csv('validation_labels_count.csv')

        self.create_figure(self.validation_score['f1_micro'], 'validation_f1_score_micro.png', 'f1-score micro')
        self.create_figure(self.validation_score['f1_macro'], 'validation_f1_score_macro.png', 'f1-score macro')
        self.create_figure(self.validation_score['loss'], 'validation_loss.png', 'loss')
        self.create_figure(self.validation_score['accuracy'], 'validation_accuracy.png', 'accuracy')

        print('Val F1-Score Micro = ', "{:.4f}".format(mean(scores['f1_micro'])), '| Val F1-Score Macro = ', "{:.4f}".format(mean(scores['f1_macro'])), '| Val Loss = ', "{:.4f}".format(mean(scores['loss'])), f'| Val Accuracy = {mean(scores["accuracy"])*100:.2f}%')

        # print('Politik = ', str(labels['politik'].count(1)))

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