import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,cohen_kappa_score,precision_score,recall_score
import numpy as np
import pandas as pd

def train_save(epoch,train_loss,pre_loss,G,P,savepath) ->None:
    acc=accuracy_score(G,P)
    f1=f1_score(G, P, average=None)
    MF1=f1_score(G, P, average='macro')
    kappa = cohen_kappa_score(G,P)

    if not os.path.exists(savepath):  
        with open(savepath, 'w') as F:
            F.write('Record:\n')
    with open(savepath, 'a') as F:
        F.write('epoch:{}__________________________\n'.format(epoch))
        F.write(f'train_loss: {train_loss}')
        F.write('\n')
        F.write(f'test_loss: {pre_loss}')
        F.write('\n')
        F.write(f'acc: {acc}')
        F.write('\n')
        F.write(f'MF1: {MF1}')
        F.write('\n')
        F.write(f'kappa: {kappa}')
        F.write('\n')
        F.write('F1-score:\n')
        F.write(str(f1))
        F.write('\n')

def retest_fold_save(G,P,filename): 
    c={"g" : G, "p" : P}
    df = pd.DataFrame(c)
    df.to_csv(filename)


def retest_save(labels,savepath):
    ## evaluation
    assert len(labels['g']) == len(labels['p'])
    G = np.array(labels['g']) 
    P = np.array(labels['p'])
    acc=accuracy_score(G,P)
    f1=f1_score(G,P,average=None)
    MF1=f1_score(G,P, average='macro')
    kappa = cohen_kappa_score(G,P)
    cm = confusion_matrix(G,P)
    precision = precision_score(G,P,average=None)
    recall = recall_score(G,P,average=None)
    if not os.path.exists(savepath):  
        with open(savepath, 'w') as F:
            F.write('Final result record:\n')
    with open(savepath, 'a') as F:    
        F.write('overal metric:\n')
        F.write(f'ACC={acc}, MF1={MF1}, Kappa={kappa}')
        F.write('\n class metric:\n')
        F.write('F1-score:')
        F.write(str(f1))
        F.write('\n precision:')
        F.write(str(precision))
        F.write('\n recall:')
        F.write(str(recall))
        F.write('\n confusion matrix:\n')   
        F.write(str(cm))
        F.write('\n')
                    









        