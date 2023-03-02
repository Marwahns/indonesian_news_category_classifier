import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

pd.options.display.float_format = '{:,.2f}'.format
training = pd.read_csv('./graph/training_scores.csv')
validation = pd.read_csv('./graph/validation_scores.csv')
training.rename(columns={training.columns[0]: 'epoch'}, inplace=True)
validation.rename(columns={validation.columns[0]: 'epoch'}, inplace=True)

def create_graph(train, val, title, fig_dir):
    plt.xlabel('epoch')
    plt.ylabel(title)
    plt.plot(training['epoch'],train, marker='o', label='Train')
    plt.plot(validation['epoch'], val, marker='o', label='Validation')
    
    # annotate_graph(train)
    # annotate_graph(val)
    plt.legend()
    plt.savefig('Training and Validation '+ fig_dir)
    plt.show()

def annotate_graph(data):
    for x_epoch, y_sc in enumerate(data):
        y_sc_lbl = '{:.2f}'.format(y_sc)

        plt.annotate(y_sc_lbl,
                      (x_epoch, y_sc),
                      textcoords='offset points',
                      xytext=(0,4),
                      ha='center')
        
create_graph(training['loss'], validation['loss'], 'Loss', 'Loss')
create_graph(training['f1_micro'], validation['f1_micro'], 'F1-Score Micro', 'F1-Score Micro')
create_graph(training['f1_macro'], validation['f1_macro'], 'F1-Score Macro', 'F1-Score Macro')
create_graph(training['accuracy'], validation['accuracy'], 'Accuracy', 'Accuracy')