import logging
import torch
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import logging
import seaborn as sns
from torch.utils.data import random_split

logging.basicConfig(encoding='utf-8')

def train_val_test_split(dataset, train_size = 0.7, valid_size = 0.15, test_size = 0.15, debug = False):
    '''
    Split the dataset into training, validate, and test dataset
    @params
        dataset: torch.utils.data.Dataset
        train_size: float = 0.7, percentage of train size.
        valid_size: float = 0.15, percentage of valid size.
        test_size: float = 0.15, percentage of test size.
        debug: boolean = False, print the result datasets size
    @return
        train_dataset: torch.utils.data.Dataset
        valid_dataset: torch.utils.data.Dataset
        test_dataset: torch.utils.data.Dataset
    '''
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
    if debug:
        print('train dataset len = ', len(train_dataset))
        print('valid dataset len = ', len(valid_dataset))
        print('test dataset len = ', len(test_dataset))
    return train_dataset, valid_dataset, test_dataset

class Trainer():
    def __init__(self, log_level = logging.DEBUG):
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        self.logger = logging.getLogger()
        self.logger.setLevel(log_level)
        self.logger.info('created trainer')

    def prepare_model(self, model, device):
        model.to(device)
        return model
    
    def prepare_batch(self, batch, device, label_type = torch.LongTensor):
        features, labels = batch
        features = features.to(device)
        labels = labels.type(label_type).to(device)
        return features, labels
    
    def score(self, score_fn, model, dataloader, device, criterion = None, name ='', **kwargs):
        y_pred = np.array([])
        y = np.array([])
        loop = tqdm(dataloader)
        loop.set_description(f'evaluating score {name}...')
        running_loss = 0.0 if criterion != None else None
        for batch in loop: 
            features, labels = self.prepare_batch(batch, device)
            outputs = model(features)

            #get loss values if criterion not None 
            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item()

            outputs = torch.argmax(outputs, dim = 1).detach().cpu().numpy().reshape(-1)
            labels = labels.detach().cpu().numpy().reshape(-1)
            y_pred = np.concatenate((y_pred, outputs))
            y = np.concatenate((y, labels))
        
        return score_fn(y_pred, y, **kwargs), running_loss 

    def confusion_matrix(self, model, dataloader, device, **kwwargs):
        y_pred = np.array([])
        y = np.array([])
        for batch in dataloader:
            features, labels = self.prepare_batch(batch, device)
            outputs = model(features)
            outputs = torch.argmax(outputs, dim = 1).detach().cpu().numpy().reshape(-1)
            labels = labels.detach().cpu().numpy().reshape(-1)
            y_pred = np.concatenate((y_pred, outputs))
            y = np.concatenate((y, labels))

        cm = metrics.confusion_matrix(y_pred, y)
        sns.heatmap(cm, annot = True)
    
    def performance_plot(self, history):
        train_loss =[]
        val_loss =[]
        test_loss =[]
        train_acc = []
        val_acc = []
        test_acc = []

        for hist in history:
            train_loss.append(hist['train_loss'])
            val_loss.append(hist['val_loss'])
            test_loss.append(hist['test_loss'])
            train_acc.append(hist['train_accuracy'])
            val_acc.append(hist['val_accuracy'])
            test_acc.append(hist['test_accuracy'])
        
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10,5))
        e = list(range(len(history)))
        
        sns.lineplot(x = e, y = train_loss, label = 'train loss', ax = ax1)
        sns.lineplot(x = e, y = val_loss, label = 'val loss', ax = ax1)
        sns.lineplot(x = e, y = test_loss, label = 'test loss', ax = ax1)
        sns.lineplot(x = e, y = train_acc, label = 'train accuracy', ax = ax2)
        sns.lineplot(x = e, y = val_acc, label = 'val accuracy', ax = ax2)
        sns.lineplot(x = e, y = test_acc, label = 'test accuracy', ax = ax2)
        plt.show()
        

    def train(self, model, optim, criterion,  train_dataloader, val_dataloader = None, \
              test_dataloader = None, device = None, epochs = 10, save_path = None):

        if device == None:
            device = self.device

        self.logger.info(f"DEVICE = {device}")

        history = []
        self.logger.debug('preparing model for training ... ')
        model = self.prepare_model(model, device)
        self.logger.debug('model prep done')

        self.logger.info('training ... ')

        best_score = 0

        for e in range(epochs):
            loop = tqdm(train_dataloader)

            total_train_loss = 0
            loop.set_description(f'training epoch {e}\t\t')

            model.train()

            for batch in loop:
                features, labels = self.prepare_batch(batch, device)

                outputs = model(features)

                optim.zero_grad()
                loss = criterion(outputs, labels)
                loss.backward()
                optim.step()

                total_train_loss += loss.item()

            model.eval()
            train_accuracy, train_loss = self.score(metrics.accuracy_score, model, train_dataloader, device, criterion=criterion, name = 'train')
            if val_dataloader != None:
                val_accuracy, val_loss = self.score(metrics.accuracy_score, model, val_dataloader, device, criterion=criterion, name = 'valid')
            else:
                val_accuracy, val_loss = None, None
            
            if test_dataloader != None:
                test_accuracy, test_loss = self.score(metrics.accuracy_score, model, test_dataloader, device, criterion=criterion, name = 'test')
            else:
                test_accuracy, test_loss = None, None


            self.logger.debug(f'\n\
                            epoch = {e}\n\
                            =============\n\
                            train_loss = {train_loss:.2f}\n\
                            train_accuracy = {train_accuracy:.2f}\n\
                            val loss = {val_loss:.2f}\n\
                            valid_accuracy = {val_accuracy:.2f}\n\
                            test loss = {test_loss:.2f}\n\
                            test_accuracy = {test_accuracy:.2f}\n\
                            bets_accuracy = {best_score:.2f}\n\
                            ')
            history.append({
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
            })
            
            if best_score < test_accuracy:
                best_score = test_accuracy

                if save_path:
                    self.logger.info(f"saved best score model at checkpoint.{save_path}")
                    torch.save(model, 'checkpoint.' + save_path)

        self.logger.info("done training ...")

        return model, history 