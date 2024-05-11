import torch
import torch.nn as nn   
from torch.utils.data import DataLoader
from datagenerator.SleepDataGene import SleepSC_Gene
from train_func import train, load_model,predict
from models.UnetModel import UnetNN
import os
from utils.result_save import retest_save,retest_fold_save


def main_process(data_rootpath,model_savepath,result_savepath,fold_num,data_source,random_state):
    dataset_split = SleepSC_Gene(fold_num,data_rootpath,data_source, random_state)  # K-fold
    retest_result = {'g':[],'p':[]}  
    # train stage ---------------------------------------------------------------------------------------------------- #
    for fold_order in range(fold_num):     
        torch.manual_seed(1)  
        # 1. set data           
        train_dataset, test_dataset = dataset_split.train_test_data(fold_order,data_rootpath)     
        # 2. set model 
        model = UnetNN(img_ch=1,num_classes=5,fc_num=2)    
        print(model)
        # 3. train model
        output_name = os.path.join(model_savepath,f'SleepSCModel_{data_source}_fold{fold_order}_best_model.pth.tar')
        record_savepath = os.path.join(result_savepath, f'fold{fold_order}_{data_source}_20fold_history.txt')
        l_rate = 0.001
        use_criterion = nn.CrossEntropyLoss()
        train(model=model, dataset=train_dataset, val_dataset=test_dataset, result_savepath = record_savepath, 
                loss_fun=use_criterion, lr=l_rate, num_epoch=100, batch_size=256, output_name=output_name
                )
        # 4. retest stage   
        model = load_model(model=model, output_name=output_name)
        val_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4) 
        G,P = predict(model, val_dataloader, loss_fun=use_criterion, arg_flag=True,train_process=False) #计算测试结果
        retest_result['g'].extend(G.tolist())
        retest_result['p'].extend(P.tolist())
        retest_fold_filename = os.path.join(result_savepath,f'retest_{data_source}_fold{fold_order}_label_result.csv')
        retest_fold_save(G.tolist(),P.tolist(),retest_fold_filename)
    # 5. save final result
    retest_savepath = os.path.join(result_savepath,f'retest_{data_source}_20fold_result.txt')
    retest_save(retest_result,retest_savepath)             

if __name__ == '__main__':
    
    model_savepath = './models/weights'  
    if not os.path.exists(model_savepath):
        os.makedirs(model_savepath)
    result_savepath = './records'  
    if not os.path.exists(result_savepath):
        os.makedirs(result_savepath)
    data_rootpath = 'D:/data/SleepSC_data'
    if not os.path.exists(data_rootpath):
        print('data source is wrong')
    main_process(data_rootpath,model_savepath,result_savepath,fold_num=20,
                 data_source = 'edf20',random_state = 1)