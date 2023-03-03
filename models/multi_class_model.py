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

        self.train_true_labels_count = {
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

        self.train_prediction_labels_count = {
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

        self.validation_true_labels_count = {
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

        self.validation_prediction_labels_count = {
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

        true_labels = {
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

        prediction_labels = {
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

            ## True Labels
            true_labels['daerah'].append(output['labels'][0][0].detach().cpu().item())
            true_labels['ekbis'].append(output['labels'][0][1].detach().cpu().item())
            true_labels['entertainment'].append(output['labels'][0][2].detach().cpu().item())
            true_labels['foto'].append(output['labels'][0][3].detach().cpu().item())
            true_labels['global'].append(output['labels'][0][4].detach().cpu().item())
            true_labels['hankam'].append(output['labels'][0][5].detach().cpu().item())
            true_labels['history'].append(output['labels'][0][6].detach().cpu().item())
            true_labels['hukum'].append(output['labels'][0][7].detach().cpu().item())
            true_labels['kesehatan'].append(output['labels'][0][8].detach().cpu().item())
            true_labels['khazanah'].append(output['labels'][0][9].detach().cpu().item())
            true_labels['lifestyle'].append(output['labels'][0][10].detach().cpu().item())
            true_labels['metro'].append(output['labels'][0][11].detach().cpu().item())
            true_labels['militer'].append(output['labels'][0][12].detach().cpu().item())
            true_labels['nasional'].append(output['labels'][0][13].detach().cpu().item())
            true_labels['otomotif'].append(output['labels'][0][14].detach().cpu().item())
            true_labels['peristiwa'].append(output['labels'][0][15].detach().cpu().item())
            true_labels['politik'].append(output['labels'][0][16].detach().cpu().item())
            true_labels['property'].append(output['labels'][0][17].detach().cpu().item())
            true_labels['seleb'].append(output['labels'][0][18].detach().cpu().item())
            true_labels['sosmed'].append(output['labels'][0][19].detach().cpu().item())
            true_labels['sport'].append(output['labels'][0][20].detach().cpu().item())
            true_labels['techno'].append(output['labels'][0][21].detach().cpu().item())

            ## Prediction Labels
            prediction_labels['daerah'].append(output['predictions'][0][0].detach().cpu().item())
            prediction_labels['ekbis'].append(output['predictions'][0][1].detach().cpu().item())
            prediction_labels['entertainment'].append(output['predictions'][0][2].detach().cpu().item())
            prediction_labels['foto'].append(output['predictions'][0][3].detach().cpu().item())
            prediction_labels['global'].append(output['predictions'][0][4].detach().cpu().item())
            prediction_labels['hankam'].append(output['predictions'][0][5].detach().cpu().item())
            prediction_labels['history'].append(output['predictions'][0][6].detach().cpu().item())
            prediction_labels['hukum'].append(output['predictions'][0][7].detach().cpu().item())
            prediction_labels['kesehatan'].append(output['predictions'][0][8].detach().cpu().item())
            prediction_labels['khazanah'].append(output['predictions'][0][9].detach().cpu().item())
            prediction_labels['lifestyle'].append(output['predictions'][0][10].detach().cpu().item())
            prediction_labels['metro'].append(output['predictions'][0][11].detach().cpu().item())
            prediction_labels['militer'].append(output['predictions'][0][12].detach().cpu().item())
            prediction_labels['nasional'].append(output['predictions'][0][13].detach().cpu().item())
            prediction_labels['otomotif'].append(output['predictions'][0][14].detach().cpu().item())
            prediction_labels['peristiwa'].append(output['predictions'][0][15].detach().cpu().item())
            prediction_labels['politik'].append(output['predictions'][0][16].detach().cpu().item())
            prediction_labels['property'].append(output['predictions'][0][17].detach().cpu().item())
            prediction_labels['seleb'].append(output['predictions'][0][18].detach().cpu().item())
            prediction_labels['sosmed'].append(output['predictions'][0][19].detach().cpu().item())
            prediction_labels['sport'].append(output['predictions'][0][20].detach().cpu().item())
            prediction_labels['techno'].append(output['predictions'][0][21].detach().cpu().item())

            ## Scores
            scores['f1_micro'].append(output['f1_micro'])
            scores['f1_macro'].append(output['f1_macro'])
            scores['loss'].append(output['loss'].detach().cpu().item())
            scores['accuracy'].append(output['accuracy'])
        
        ## True Labels Count
        self.train_true_labels_count['daerah'].append(true_labels['daerah'].count(1))
        self.train_true_labels_count['ekbis'].append(true_labels['ekbis'].count(1))
        self.train_true_labels_count['entertainment'].append(true_labels['entertainment'].count(1))
        self.train_true_labels_count['foto'].append(true_labels['foto'].count(1))
        self.train_true_labels_count['global'].append(true_labels['global'].count(1))
        self.train_true_labels_count['hankam'].append(true_labels['hankam'].count(1))
        self.train_true_labels_count['history'].append(true_labels['history'].count(1))
        self.train_true_labels_count['hukum'].append(true_labels['hukum'].count(1))
        self.train_true_labels_count['kesehatan'].append(true_labels['kesehatan'].count(1))
        self.train_true_labels_count['khazanah'].append(true_labels['khazanah'].count(1))
        self.train_true_labels_count['lifestyle'].append(true_labels['lifestyle'].count(1))
        self.train_true_labels_count['metro'].append(true_labels['metro'].count(1))
        self.train_true_labels_count['militer'].append(true_labels['militer'].count(1))
        self.train_true_labels_count['nasional'].append(true_labels['nasional'].count(1))
        self.train_true_labels_count['otomotif'].append(true_labels['otomotif'].count(1))
        self.train_true_labels_count['peristiwa'].append(true_labels['peristiwa'].count(1))
        self.train_true_labels_count['politik'].append(true_labels['politik'].count(1))
        self.train_true_labels_count['property'].append(true_labels['property'].count(1))
        self.train_true_labels_count['seleb'].append(true_labels['seleb'].count(1))
        self.train_true_labels_count['sosmed'].append(true_labels['sosmed'].count(1))
        self.train_true_labels_count['sport'].append(true_labels['sport'].count(1))
        self.train_true_labels_count['techno'].append(true_labels['techno'].count(1))
        
        ## Prediction Labels Count
        self.train_prediction_labels_count['daerah'].append(prediction_labels['daerah'].count(1))
        self.train_prediction_labels_count['ekbis'].append(prediction_labels['ekbis'].count(1))
        self.train_prediction_labels_count['entertainment'].append(prediction_labels['entertainment'].count(1))
        self.train_prediction_labels_count['foto'].append(prediction_labels['foto'].count(1))
        self.train_prediction_labels_count['global'].append(prediction_labels['global'].count(1))
        self.train_prediction_labels_count['hankam'].append(prediction_labels['hankam'].count(1))
        self.train_prediction_labels_count['history'].append(prediction_labels['history'].count(1))
        self.train_prediction_labels_count['hukum'].append(prediction_labels['hukum'].count(1))
        self.train_prediction_labels_count['kesehatan'].append(prediction_labels['kesehatan'].count(1))
        self.train_prediction_labels_count['khazanah'].append(prediction_labels['khazanah'].count(1))
        self.train_prediction_labels_count['lifestyle'].append(prediction_labels['lifestyle'].count(1))
        self.train_prediction_labels_count['metro'].append(prediction_labels['metro'].count(1))
        self.train_prediction_labels_count['militer'].append(prediction_labels['militer'].count(1))
        self.train_prediction_labels_count['nasional'].append(prediction_labels['nasional'].count(1))
        self.train_prediction_labels_count['otomotif'].append(prediction_labels['otomotif'].count(1))
        self.train_prediction_labels_count['peristiwa'].append(prediction_labels['peristiwa'].count(1))
        self.train_prediction_labels_count['politik'].append(prediction_labels['politik'].count(1))
        self.train_prediction_labels_count['property'].append(prediction_labels['property'].count(1))
        self.train_prediction_labels_count['seleb'].append(prediction_labels['seleb'].count(1))
        self.train_prediction_labels_count['sosmed'].append(prediction_labels['sosmed'].count(1))
        self.train_prediction_labels_count['sport'].append(prediction_labels['sport'].count(1))
        self.train_prediction_labels_count['techno'].append(prediction_labels['techno'].count(1))

        ## Scores
        self.train_score['f1_micro'].append(mean(scores['f1_micro']))
        self.train_score['f1_macro'].append(mean(scores['f1_macro']))
        self.train_score['loss'].append(mean(scores['loss']))
        self.train_score['accuracy'].append(mean(scores['accuracy']))

        df_scores = pd.DataFrame.from_dict(self.train_score)
        df_true_labels_count = pd.DataFrame.from_dict(self.train_true_labels_count)
        df_prediction_labels_count = pd.DataFrame.from_dict(self.train_prediction_labels_count)

        df_scores.to_csv('training_scores.csv')
        df_true_labels_count.to_csv('training_true_labels_count.csv')
        df_prediction_labels_count.to_csv('training_prediction_labels_count.csv')

        self.create_figure(self.train_score['f1_micro'], 'training_f1_score_micro.png', 'f1-score micro')
        self.create_figure(self.train_score['f1_macro'], 'training_f1_score_macro.png', 'f1-score macro')
        self.create_figure(self.train_score['loss'], 'training_loss.png', 'loss')
        self.create_figure(self.train_score['accuracy'], 'training_accuracy.png', 'accuracy')

        print('F1-Score Micro = ', "{:.4f}".format(mean(scores['f1_micro'])), 
              '| F1-Score Macro = ', "{:.4f}".format(mean(scores['f1_macro'])), 
              '| Loss = ', "{:.4f}".format(mean(scores['loss'])), 
              f'| Accuracy = {mean(scores["accuracy"])*100:.2f}%')

        ## print "{:.2f}".format(56.455323)
        ## print(f'accuracy: {a*100:.2f}%')
    
    def validation_epoch_end(self, outputs):
        scores = {
            'loss': [],
            'f1_micro': [],
            'f1_macro': [],
            'accuracy': []
        }

        true_labels = {
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

        prediction_labels = {
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
            ## True Labels
            true_labels['daerah'].append(output['labels'][0][0].detach().cpu().item())
            true_labels['ekbis'].append(output['labels'][0][1].detach().cpu().item())
            true_labels['entertainment'].append(output['labels'][0][2].detach().cpu().item())
            true_labels['foto'].append(output['labels'][0][3].detach().cpu().item())
            true_labels['global'].append(output['labels'][0][4].detach().cpu().item())
            true_labels['hankam'].append(output['labels'][0][5].detach().cpu().item())
            true_labels['history'].append(output['labels'][0][6].detach().cpu().item())
            true_labels['hukum'].append(output['labels'][0][7].detach().cpu().item())
            true_labels['kesehatan'].append(output['labels'][0][8].detach().cpu().item())
            true_labels['khazanah'].append(output['labels'][0][9].detach().cpu().item())
            true_labels['lifestyle'].append(output['labels'][0][10].detach().cpu().item())
            true_labels['metro'].append(output['labels'][0][11].detach().cpu().item())
            true_labels['militer'].append(output['labels'][0][12].detach().cpu().item())
            true_labels['nasional'].append(output['labels'][0][13].detach().cpu().item())
            true_labels['otomotif'].append(output['labels'][0][14].detach().cpu().item())
            true_labels['peristiwa'].append(output['labels'][0][15].detach().cpu().item())
            true_labels['politik'].append(output['labels'][0][16].detach().cpu().item())
            true_labels['property'].append(output['labels'][0][17].detach().cpu().item())
            true_labels['seleb'].append(output['labels'][0][18].detach().cpu().item())
            true_labels['sosmed'].append(output['labels'][0][19].detach().cpu().item())
            true_labels['sport'].append(output['labels'][0][20].detach().cpu().item())
            true_labels['techno'].append(output['labels'][0][21].detach().cpu().item())
            
            ## Prediction Labels
            prediction_labels['daerah'].append(output['predictions'][0][0].detach().cpu().item())
            prediction_labels['ekbis'].append(output['predictions'][0][1].detach().cpu().item())
            prediction_labels['entertainment'].append(output['predictions'][0][2].detach().cpu().item())
            prediction_labels['foto'].append(output['predictions'][0][3].detach().cpu().item())
            prediction_labels['global'].append(output['predictions'][0][4].detach().cpu().item())
            prediction_labels['hankam'].append(output['predictions'][0][5].detach().cpu().item())
            prediction_labels['history'].append(output['predictions'][0][6].detach().cpu().item())
            prediction_labels['hukum'].append(output['predictions'][0][7].detach().cpu().item())
            prediction_labels['kesehatan'].append(output['predictions'][0][8].detach().cpu().item())
            prediction_labels['khazanah'].append(output['predictions'][0][9].detach().cpu().item())
            prediction_labels['lifestyle'].append(output['predictions'][0][10].detach().cpu().item())
            prediction_labels['metro'].append(output['predictions'][0][11].detach().cpu().item())
            prediction_labels['militer'].append(output['predictions'][0][12].detach().cpu().item())
            prediction_labels['nasional'].append(output['predictions'][0][13].detach().cpu().item())
            prediction_labels['otomotif'].append(output['predictions'][0][14].detach().cpu().item())
            prediction_labels['peristiwa'].append(output['predictions'][0][15].detach().cpu().item())
            prediction_labels['politik'].append(output['predictions'][0][16].detach().cpu().item())
            prediction_labels['property'].append(output['predictions'][0][17].detach().cpu().item())
            prediction_labels['seleb'].append(output['predictions'][0][18].detach().cpu().item())
            prediction_labels['sosmed'].append(output['predictions'][0][19].detach().cpu().item())
            prediction_labels['sport'].append(output['predictions'][0][20].detach().cpu().item())
            prediction_labels['techno'].append(output['predictions'][0][21].detach().cpu().item())

            ## scores
            scores['f1_micro'].append(output['f1_micro'])
            scores['f1_macro'].append(output['f1_macro'])
            scores['loss'].append(output['val_loss'].detach().cpu().item())
            scores['accuracy'].append(output['accuracy'])
        
        ## True Labels Count
        self.validation_true_labels_count['daerah'].append(true_labels['daerah'].count(1))
        self.validation_true_labels_count['ekbis'].append(true_labels['ekbis'].count(1))
        self.validation_true_labels_count['entertainment'].append(true_labels['entertainment'].count(1))
        self.validation_true_labels_count['foto'].append(true_labels['foto'].count(1))
        self.validation_true_labels_count['global'].append(true_labels['global'].count(1))
        self.validation_true_labels_count['hankam'].append(true_labels['hankam'].count(1))
        self.validation_true_labels_count['history'].append(true_labels['history'].count(1))
        self.validation_true_labels_count['hukum'].append(true_labels['hukum'].count(1))
        self.validation_true_labels_count['kesehatan'].append(true_labels['kesehatan'].count(1))
        self.validation_true_labels_count['khazanah'].append(true_labels['khazanah'].count(1))
        self.validation_true_labels_count['lifestyle'].append(true_labels['lifestyle'].count(1))
        self.validation_true_labels_count['metro'].append(true_labels['metro'].count(1))
        self.validation_true_labels_count['militer'].append(true_labels['militer'].count(1))
        self.validation_true_labels_count['nasional'].append(true_labels['nasional'].count(1))
        self.validation_true_labels_count['otomotif'].append(true_labels['otomotif'].count(1))
        self.validation_true_labels_count['peristiwa'].append(true_labels['peristiwa'].count(1))
        self.validation_true_labels_count['politik'].append(true_labels['politik'].count(1))
        self.validation_true_labels_count['property'].append(true_labels['property'].count(1))
        self.validation_true_labels_count['seleb'].append(true_labels['seleb'].count(1))
        self.validation_true_labels_count['sosmed'].append(true_labels['sosmed'].count(1))
        self.validation_true_labels_count['sport'].append(true_labels['sport'].count(1))
        self.validation_true_labels_count['techno'].append(true_labels['techno'].count(1))

        ## Prediction Labels Count
        self.validation_prediction_labels_count['daerah'].append(prediction_labels['daerah'].count(1))
        self.validation_prediction_labels_count['ekbis'].append(prediction_labels['ekbis'].count(1))
        self.validation_prediction_labels_count['entertainment'].append(prediction_labels['entertainment'].count(1))
        self.validation_prediction_labels_count['foto'].append(prediction_labels['foto'].count(1))
        self.validation_prediction_labels_count['global'].append(prediction_labels['global'].count(1))
        self.validation_prediction_labels_count['hankam'].append(prediction_labels['hankam'].count(1))
        self.validation_prediction_labels_count['history'].append(prediction_labels['history'].count(1))
        self.validation_prediction_labels_count['hukum'].append(prediction_labels['hukum'].count(1))
        self.validation_prediction_labels_count['kesehatan'].append(prediction_labels['kesehatan'].count(1))
        self.validation_prediction_labels_count['khazanah'].append(prediction_labels['khazanah'].count(1))
        self.validation_prediction_labels_count['lifestyle'].append(prediction_labels['lifestyle'].count(1))
        self.validation_prediction_labels_count['metro'].append(prediction_labels['metro'].count(1))
        self.validation_prediction_labels_count['militer'].append(prediction_labels['militer'].count(1))
        self.validation_prediction_labels_count['nasional'].append(prediction_labels['nasional'].count(1))
        self.validation_prediction_labels_count['otomotif'].append(prediction_labels['otomotif'].count(1))
        self.validation_prediction_labels_count['peristiwa'].append(prediction_labels['peristiwa'].count(1))
        self.validation_prediction_labels_count['politik'].append(prediction_labels['politik'].count(1))
        self.validation_prediction_labels_count['property'].append(prediction_labels['property'].count(1))
        self.validation_prediction_labels_count['seleb'].append(prediction_labels['seleb'].count(1))
        self.validation_prediction_labels_count['sosmed'].append(prediction_labels['sosmed'].count(1))
        self.validation_prediction_labels_count['sport'].append(prediction_labels['sport'].count(1))
        self.validation_prediction_labels_count['techno'].append(prediction_labels['techno'].count(1))

        ## Scores
        self.validation_score['f1_micro'].append(mean(scores['f1_micro']))
        self.validation_score['f1_macro'].append(mean(scores['f1_macro']))
        self.validation_score['loss'].append(mean(scores['loss']))
        self.validation_score['accuracy'].append(mean(scores['accuracy']))

        df_scores = pd.DataFrame.from_dict(self.validation_score)
        df_true_labels_count = pd.DataFrame.from_dict(self.validation_true_labels_count)
        df_prediction_labels_count = pd.DataFrame.from_dict(self.validation_prediction_labels_count)

        df_scores.to_csv('validation_scores.csv')
        df_true_labels_count.to_csv('validation_true_labels_count.csv')
        df_prediction_labels_count.to_csv('validation_prediction_labels_count.csv')

        self.create_figure(self.validation_score['f1_micro'], 'validation_f1_score_micro.png', 'f1-score micro')
        self.create_figure(self.validation_score['f1_macro'], 'validation_f1_score_macro.png', 'f1-score macro')
        self.create_figure(self.validation_score['loss'], 'validation_loss.png', 'loss')
        self.create_figure(self.validation_score['accuracy'], 'validation_accuracy.png', 'accuracy')

        print('Val F1-Score Micro = ', "{:.4f}".format(mean(scores['f1_micro'])), 
              '| Val F1-Score Macro = ', "{:.4f}".format(mean(scores['f1_macro'])), 
              '| Val Loss = ', "{:.4f}".format(mean(scores['loss'])), 
              f'| Val Accuracy = {mean(scores["accuracy"])*100:.2f}%')

    def test_epoch_end(self, outputs):
        scores = {
            'f1_micro': [],
            'f1_macro': [],
            'accuracy': []
        }

        true_labels_count = {
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

        predictions_labels_count = {
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

        predictions_labels = []
        true_labels = []

        for output in outputs:
            # print(output)
                
            ## Scores
            scores['f1_micro'].append(output['f1_micro'])
            scores['f1_macro'].append(output['f1_macro'])
            scores['accuracy'].append(output['accuracy'])

            ## True Labels
            for true in output['labels'].detach().cpu().numpy():
                # print(true)
                true_labels.append(true)
            
            ## Predict Labels
            for pred in output['predictions'].detach().cpu().numpy():
                predictions_labels.append(pred)
        
        ## True Labels Count
        true_labels_count['daerah'].append(true_labels.count(0))
        true_labels_count['ekbis'].append(true_labels.count(1))
        true_labels_count['entertainment'].append(true_labels.count(2))
        true_labels_count['foto'].append(true_labels.count(3))
        true_labels_count['global'].append(true_labels.count(4))
        true_labels_count['hankam'].append(true_labels.count(5))
        true_labels_count['history'].append(true_labels.count(6))
        true_labels_count['hukum'].append(true_labels.count(7))
        true_labels_count['kesehatan'].append(true_labels.count(8))
        true_labels_count['khazanah'].append(true_labels.count(9))
        true_labels_count['lifestyle'].append(true_labels.count(10))
        true_labels_count['metro'].append(true_labels.count(11))
        true_labels_count['militer'].append(true_labels.count(12))
        true_labels_count['nasional'].append(true_labels.count(13))
        true_labels_count['otomotif'].append(true_labels.count(14))
        true_labels_count['peristiwa'].append(true_labels.count(15))
        true_labels_count['politik'].append(true_labels.count(16))
        true_labels_count['property'].append(true_labels.count(17))
        true_labels_count['seleb'].append(true_labels.count(18))
        true_labels_count['sosmed'].append(true_labels.count(19))
        true_labels_count['sport'].append(true_labels.count(20))
        true_labels_count['techno'].append(true_labels.count(21))

        ## Prediction Labels Count
        predictions_labels_count['daerah'].append(predictions_labels.count(0))
        predictions_labels_count['ekbis'].append(predictions_labels.count(1))
        predictions_labels_count['entertainment'].append(predictions_labels.count(2))
        predictions_labels_count['foto'].append(predictions_labels.count(3))
        predictions_labels_count['global'].append(predictions_labels.count(4))
        predictions_labels_count['hankam'].append(predictions_labels.count(5))
        predictions_labels_count['history'].append(predictions_labels.count(6))
        predictions_labels_count['hukum'].append(predictions_labels.count(7))
        predictions_labels_count['kesehatan'].append(predictions_labels.count(8))
        predictions_labels_count['khazanah'].append(predictions_labels.count(9))
        predictions_labels_count['lifestyle'].append(predictions_labels.count(10))
        predictions_labels_count['metro'].append(predictions_labels.count(11))
        predictions_labels_count['militer'].append(predictions_labels.count(12))
        predictions_labels_count['nasional'].append(predictions_labels.count(13))
        predictions_labels_count['otomotif'].append(predictions_labels.count(14))
        predictions_labels_count['peristiwa'].append(predictions_labels.count(15))
        predictions_labels_count['politik'].append(predictions_labels.count(16))
        predictions_labels_count['property'].append(predictions_labels.count(17))
        predictions_labels_count['seleb'].append(predictions_labels.count(18))
        predictions_labels_count['sosmed'].append(predictions_labels.count(19))
        predictions_labels_count['sport'].append(predictions_labels.count(20))
        predictions_labels_count['techno'].append(predictions_labels.count(21))
        
        ## Scores
        print('F1-Score Micro = ', mean(scores['f1_micro']), 
              '| F1-Score Macro = ', mean(scores['f1_macro']), 
              f'accuracy: {mean(scores["accuracy"])*100:.2f}%')
        
        ## Display True Labels
        print('\nTrue Labels')
        print('Daerah = ', true_labels_count['daerah'])
        print('Ekbis = ', true_labels_count['ekbis'])
        print('Entertainment = ', true_labels_count['entertainment'])
        print('Foto = ', true_labels_count['foto'])
        print('Global = ', true_labels_count['global'])
        print('Hankam = ', true_labels_count['hankam'])
        print('History = ', true_labels_count['history'])
        print('Hukum = ', true_labels_count['hukum'])
        print('Kesehatan = ', true_labels_count['kesehatan'])
        print('Khazanah = ', true_labels_count['khazanah'])
        print('Lifestyle = ', true_labels_count['lifestyle'])
        print('Metro = ', true_labels_count['metro'])
        print('Militer = ', true_labels_count['militer'])
        print('Nasional = ', true_labels_count['nasional'])
        print('Otomotif = ', true_labels_count['otomotif'])
        print('Peristiwa = ', true_labels_count['peristiwa'])
        print('Politik = ', true_labels_count['politik'])
        print('Property = ', true_labels_count['property'])
        print('Seleb = ', true_labels_count['seleb'])
        print('Sosmed = ', true_labels_count['sosmed'])
        print('Sport = ', true_labels_count['sport'])
        print('Techno = ', true_labels_count['techno'])

        ## Display Prediction Labels
        print('\nPrediction Labels')
        print('Daerah = ', predictions_labels_count['daerah'])
        print('Ekbis = ', predictions_labels_count['ekbis'])
        print('Entertainment = ', predictions_labels_count['entertainment'])
        print('Foto = ', predictions_labels_count['foto'])
        print('Global = ', predictions_labels_count['global'])
        print('Hankam = ', predictions_labels_count['hankam'])
        print('History = ', predictions_labels_count['history'])
        print('Hukum = ', predictions_labels_count['hukum'])
        print('Kesehatan = ', predictions_labels_count['kesehatan'])
        print('Khazanah = ', predictions_labels_count['khazanah'])
        print('Lifestyle = ', predictions_labels_count['lifestyle'])
        print('Metro = ', predictions_labels_count['metro'])
        print('Militer = ', predictions_labels_count['militer'])
        print('Nasional = ', predictions_labels_count['nasional'])
        print('Otomotif = ', predictions_labels_count['otomotif'])
        print('Peristiwa = ', predictions_labels_count['peristiwa'])
        print('Politik = ', predictions_labels_count['politik'])
        print('Property = ', predictions_labels_count['property'])
        print('Seleb = ', predictions_labels_count['seleb'])
        print('Sosmed = ', predictions_labels_count['sosmed'])
        print('Sport = ', predictions_labels_count['sport'])
        print('Techno = ', predictions_labels_count['techno'])

        df_true_labels = pd.DataFrame.from_dict(true_labels_count)
        df_prediction_labels = pd.DataFrame.from_dict(predictions_labels_count)

        df_true_labels.to_csv('test_true_labels.csv')
        df_prediction_labels.to_csv('test_prediction_labels.csv')

        ## output
        ## F1-Score Micro =  0.06022727272727274 
        ## F1-Score Macro =  0.02574716472951051 
        ## accuracy: 6.02%