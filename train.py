import torch
import torch.nn as nn
from torch.nn import functional as F
from time import time

class TrainingModule:
    def __init__(self, model, optimizer, criterion, train_loader, val_loader, epochs, scheduler = None, checkpoint = True, early_stop = False):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.scheduler = scheduler
        self.checkpoint = checkpoint
        self.early_stop = early_stop
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def train(self):
        list_train_loss = []
        list_val_loss = []
        list_train_error = []
        list_val_error = []
        
        best_val_loss = float('+inf')
        
        self.model.to(self.DEVICE)
        self.criterion = self.criterion.to(self.DEVICE)
        
        for epoch in range(self.epochs):
          
            start_time = time()

            train_loss, train_error = self.run_epoch(dataloader = self.train_loader, mode='train')
            val_loss, val_error = self.run_epoch(dataloader = self.val_loader, mode='test')

            list_train_loss.append(train_loss)
            list_val_loss.append(val_loss)
            list_train_error.append(train_error)
            list_val_error.append(val_error)

            # checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                if self.checkpoint:
                    torch.save(self.model.state_dict(), './best_model.pth')
            
            if self.scheduler is not None:
                self.scheduler.step(train_loss)

            if (epoch+1) % 1 == 0:
              print('Epoch {}: train loss - {}, train error - {}'.format(epoch+1, round(train_loss,5), train_error))
              print('Epoch {}: val loss - {}, val error - {}'.format(epoch+1, round(val_loss,5), val_error))
              print ('Elapsed time: %.3f' % (time() - start_time))
              print('----------------------------------------------------------------------')
            
        return list_train_loss, list_val_loss, list_train_error, list_val_error


    def run_epoch(self, mode, dataloader):
      
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        epoch_loss = 0.0
        
        num_batches = 0
        for vector, target in dataloader:
            vector = vector.to(self.DEVICE)
            target = target.to(self.DEVICE)
            
            prediction = self.model(vector)
            loss = self.criterion(prediction, target)
            
            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            
            if self.early_stop:
                break
        
        num_batches = float(num_batches)

        return epoch_loss/num_batches, epoch_error/num_batches
