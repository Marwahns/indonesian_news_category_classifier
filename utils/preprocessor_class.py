import enum
import pickle
import torch
import os

import re

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import pytorch_lightning as pl
import pandas as pd

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import BertTokenizer

# untuk membuat progress bar
from tqdm import tqdm 

class PreprocessorClass(pl.LightningDataModule):

    def __init__(self,
                 preprocessed_dir,
                 ## 'news_multi_class_classification/data/train.csv',
                 train_data_dir = './data/train.csv',
                 test_data_dir = './data/test.csv',
                 batch_size = 10,
                 max_length = 100):
        
        super(PreprocessorClass, self).__init__()

        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir

        ## load data file csv
        # news_data = pd.read_csv('./datasets/data.csv')
        # print(news_data.columns)

        ## konversi label ke huruf kecil
        # news_data['label'] = news_data['label'].str.lower()
        # news_data_groupby_label = news_data.groupby("label")

        ## jumlah masing-masing label 
        # news_data_groupby_label["label"].count()

        # news_data_raw = news_data.rename(columns={1: "judul", 2: "label", 3: "isi_berita", 4: "url"})

        ## inisialisai isi label menjadi id
        self.label2id = {
            'daerah': 0,
            'ekbis': 1,
            'entertainment': 2,
            'foto': 3,
            'global': 4,
            'hankam': 5,
            'history': 6,
            'hukum': 7,
            'kesehatan': 8,
            'khazanah': 9,
            'lifestyle': 10,
            'metro': 11,
            'militer': 12,
            'nasional': 13,
            'otomotif': 14,
            'peristiwa': 15,
            'politik': 16,
            'property': 17,
            'seleb': 18,
            'sosmed': 19,
            'sport': 20,
            'techno': 21
        }

        ## konversi label menjadi id
        # news_data_raw.label =  news_data_raw.label.map(self.labelid)
        # news_data_groupby_label_after = news_data_raw.groupby("label")
        # print(news_data_groupby_label_after["label"].count())

        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()

        # Merubah kalimat menjadi id, tokenize dan attention
        self.tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')

        self.max_length = max_length
        self.preprocessed_dir = preprocessed_dir

        self.batch_size = batch_size
    
    def clean_str(self, string):
        string = string.lower()
        string = re.sub(r"[^A-Za-z0-9(),!?\'\-`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\n", "", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = string.strip()

        return self.stemmer.stem(string)
    
    def load_data(self,):
        with open(self.train_data_dir, "rb") as tdr:
            train_csv = pd.read_csv(tdr)
            train_csv['label'] = train_csv['label'].str.lower()
            train = pd.DataFrame({'judul': train_csv['judul'], 'label': train_csv['label'], 'isi_berita': train_csv['isi_berita'], 'url': train_csv['url']})
        with open(self.test_data_dir, "rb") as tsdr:
            test_csv = pd.read_csv(tsdr)
            test_csv['label'] = test_csv['label'].str.lower()
            test = pd.DataFrame({'judul':test_csv['judul'], 'label': test_csv['label'], 'isi_berita': test_csv['isi_berita'], 'url': test_csv['url']})
    
        ## Mengetahui apa saja label yang ada di dalam dataset
        label_yang_ada = train["label"].drop_duplicates()
        # print(label_yang_ada)
        
        ## Konversi dari label text (news) ke label id (1)
        train.label = train.label.map(self.label2id)

        ## Dilakukan pengecekan terhadap data yang NaN
        # print(train.isna().sum())

        ## Delete baris yang berisi data NaN
        train = train.dropna()

        ## Dilakukan pengecekan terhadap data yang null
        # print(train.isnull().sum())

        ## Konversi tipe data float ke integer
        train['label'] = train['label'].astype(int)
        # print(train["label"].drop_duplicates())

        test.label = test.label.map(self.label2id)
        # print(test.isna().sum())
        test = test.dropna()
        # print(test.isnull().sum())
        test['label'] = test['label'].astype(int)
        # print(train["label"].drop_duplicates())

        return train, test
    
    def arrange_data(self, data, type):
        ## Yang di lakukan
        ## 1. Cleaning sentence
        ## 2. Tokenizing data
        ## 3. Arrange ke dataset (training, validation, testing)
        ## type merupakan tipe datanya, apakah training atau tessting

        ## y = label
        x_input_ids, x_token_type_ids, x_attention_mask, y = [], [], [], []

        ## baris untuk indexnya, index keberapa
        for baris, dt in enumerate(tqdm(data.values.tolist())):
            title = self.clean_str(dt[0])
            label = dt[1]

            ## Mengubah label yang tadinya angka menjadi binary
            binary_lbl = [0] * len(self.label2id)
            binary_lbl[label] = 1

            tkn = self.tokenizer(text = title,
                                 max_length = self.max_length,
                                 padding = "max_length",
                                 truncation = True)
            
            x_input_ids.append(tkn['input_ids'])
            x_token_type_ids.append(tkn['token_type_ids'])
            x_attention_mask.append(tkn['attention_mask'])
            y.append(binary_lbl)

        ## Mengubah list ke tensor
        x_input_ids = torch.tensor(x_input_ids)
        x_token_type_ids = torch.tensor(x_token_type_ids)
        x_attention_mask = torch.tensor(x_attention_mask)
        y = torch.tensor(y)

        tensor_dataset = TensorDataset(x_input_ids,
                                       x_token_type_ids,
                                       x_attention_mask,
                                       y)
        
        ## Memisahkan dua data (testing, dan training). Memisahkan testing menjadi validation
        if type == "train":
            ## split secara random
            ## Standard split: Train (80%), Validation (20%)
            train_tensor_dataset, valid_tensor_dataset = torch.utils.data.random_split(tensor_dataset, [
            round(len(x_input_ids) * 0.8),
            len(x_input_ids) - round(len(x_input_ids) * 0.8)
            ])

            ## f untuk merubah ke string
            torch.save(train_tensor_dataset, f"{self.preprocessed_dir}/train.pt")
            torch.save(valid_tensor_dataset, f"{self.preprocessed_dir}/valid.pt")

            return train_tensor_dataset, valid_tensor_dataset

        else:
            torch.save(tensor_dataset, f"{self.preprocessed_dir}/test.pt")


    def preprocessor(self,):
        ## membersihkan dan membuat tokenisasi
        train, test = self.load_data()

        ## Menggabungkan string yang ada di 
        ## Mengecek apakah data train dan valid sudah di preprocessed
        if not os.path.exists(f"{self.preprocessed_dir}/train.pt") or not os.path.exists(f"{self.preprocessed_dir}/valid.pt"):
            print("Create Train and Validation dataset")
            train_data, valid_data = self.arrange_data(data = train, type = "train")
        else:
            print("Load Preprocessed train and validation data")
            train_data = torch.load(f"{self.preprocessed_dir}/train.pt")
            valid_data = torch.load(f"{self.preprocessed_dir}/valid.pt")
        
        ## Mengecek apakah data testnya sudah di preprocessed
        if not os.path.exists(f"{self.preprocessed_dir}/test.pt"):
            print("Create test dataset")
            test_data = self.arrange_data(data = test, type = "test")
        else:
            print("Load Preprocessed test data")
            test_data = torch.load(f"{self.preprocessed_dir}/test.pt")

        return train_data, valid_data, test_data

    def setup(self, stage = None):
        train_data, valid_data, test_data = self.preprocessor()
        ## fit = training
        ## test = testing
        if stage == "fit":
            self.train_data = train_data
            self.valid_data = valid_data
        elif stage == "test":
            self.test_data = test_data
        
    def train_dataloader(self):
        sampler = RandomSampler(self.train_data)
        return DataLoader(
            dataset = self.train_data,
            # Membagi process training dalam sekali proses
            batch_size = self.batch_size, 
            sampler = sampler,
            num_workers = 1
        )
    
    def val_dataloader(self):
        sampler = SequentialSampler(self.valid_data)
        return DataLoader(
            dataset = self.valid_data,
            # Membagi process training dalam sekali proses
            batch_size = self.batch_size, 
            sampler = sampler,
            num_workers = 1
        )
    
    def test_dataloader(self):
        sampler = SequentialSampler(self.test_data)
        return DataLoader(
            dataset = self.test_data,
            ## Membagi process training dalam sekali proses
            batch_size = self.batch_size, 
            sampler = sampler,
            num_workers = 1
        )

if __name__ == '__main__':
    Pre = PreprocessorClass(preprocessed_dir = "./data/preprocessed")
    Pre.setup(stage = "fit")
    train_data = Pre.train_dataloader()
    print(train_data)
