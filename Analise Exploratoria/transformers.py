import torch
from torch.utils.data import Dataset

# DATASET CLASS
class Dataset(Dataset):

    def __init__(self, dataframe, y_cols):
        
        self.data = dataframe
        self.y_cols = y_cols


    def __len__(self):
        return len(self.data)


    def __getitem__(self, i):

        X = self.data.drop(self.y_cols, axis=1).iloc[i].values
        Y = self.data[self.y_cols].iloc[i]
       
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        
        return X, Y
    










from device import cuda
from checkpoint import save
from utils import results
import torch
import pandas as pd

class rate():
    
    def __init__(self, model, opt, loss):
        
        print(cuda().device_name)
        self.device = cuda().device
        self.model = model
        self.opt = opt
        self.loss = loss


    def train(self, epochs, df_train, save_model=True):

        self.model.train()

        self.train_results = {'epochs':[], 'loss':[]}

        for epoch in range(epochs+1):
            
            train_loss = 0.0
            corrects = 0
            values = 0
            
            for x, y in df_train:
            
                self.opt.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model.forward(x).to(self.device)
                lossfunc = self.loss(output, y)
                lossfunc.backward()
                self.opt.step()
                train_loss += lossfunc.item()
                corrects += int(torch.sum((torch.sum((((output == torch.max(output)).float() == y).float()), dim=1) == y.shape[1]).float(), dim=0).tolist())
                values += len(output)

            self.train_results['epochs'].append(epoch)
            self.train_results['loss'].append(train_loss/len(df_train))
                
            if epoch % 10 == 0:

                if save_model: save(model=self.model, optimizer=self.opt)
                
                load = f'{epoch/epochs*100:.2f}'
                loss = f'{train_loss/len(df_train):.2f}'
                acc = f'{corrects/values*100:.2f}'

                results(values={'Epoch':[epoch, 15]
                               ,'Load%':[load, 15]
                               ,'Loss':[loss, 15]
                               ,'Acc':[acc, 15]}
                        ,title='TRAIN RESULTS'
                        ,row=epoch)
        
        self.train_results = pd.DataFrame(self.train_results)
        
        return self
                

    def test(self, df_test):
        
        self.model.eval()
        test_loss = 0.0
        
        predicted_test = {'x':[], 'y':[], 'pred':[]}

        with torch.no_grad():
        
            corrects = 0
            values = 0
        
            for x, y in df_test:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model.forward(x).to(self.device)
                test_loss += self.loss(output, y).item()
                
                # SAVE PREDICT MODEL
                predicted_test['x'].append(x.tolist())
                predicted_test['y'].append(y.tolist())
                predicted_test['pred'].append((output == torch.max(output)).float().tolist())              

                # ACCURACY 
                values += len(output)
                corrects += int(torch.sum((torch.sum((((output == torch.max(output)).float() == y).float()), dim=1) == y.shape[1]).float(), dim=0).tolist())
        
        results(values={'Epoch':[0, 15]
                       ,'Load%':[100, 15]
                       ,'Loss':[round(test_loss/len(df_test),4), 15]
                       ,'Acc':[round(corrects/values,4)*100, 15]
                       }
                               
                       ,title='TEST RESULTS'
                       ,row=0)
        
        self.predicted_test = predicted_test
        
        return self
    

    def predict(self, df_predict):


        self.model.eval()
        
        predicted = {'x':[], 'pred':[]}

        with torch.no_grad():
        
            corrects = 0
            values = 0
        
            for x, _ in df_predict:

                x = x.to(self.device)
                output = self.model.forward(x).to(self.device)
                
                # SAVE PREDICT MODEL
                predicted['x'].append(x.tolist())
                predicted['pred'].append((output == torch.max(output)).float().tolist())              

                # ACCURACY 
                values += len(output)
                corrects += int(torch.sum((torch.sum((((output == torch.max(output)).float() == y).float()), dim=1) == y.shape[1]).float(), dim=0).tolist())
        
        self.predicted = predicted
        
        return self
    









from torch import nn

class teste1(nn.Module):

    def __init__(self, n, y):

        super().__init__()

        self.layers = nn.Sequential(
                      nn.Linear(n, 8),
                      nn.ReLU(),
                      nn.Linear(8, y),
                      nn.Softmax()
                      )
        
    def forward(self, x):
        
        output = self.layers(x)
        return output
    







from torch.utils.data import DataLoader

# DATALOADER
self.train_dataloader = DataLoader(self.train_dataset, batch_size=100, shuffle=True)
self.test_dataloader = DataLoader(self.test_dataset)







from torch import save as s


class save():

    def __init__(self, model, optimizer):
        
        s(model.state_dict(), 'model.pth')
        s(optimizer.state_dict(), 'optimizer.pth')












from torch import cuda as c

class cuda():
    
    def __init__(self):
    
        self.is_available = c.is_available()
        self.device = 'cuda' if c.is_available() else 'cpu'
        self.device_name = c.get_device_name(c.current_device()) if c.is_available() else 'CPU'







import pandas as pd

class results():
    
    def __init__(self, values, title, row):
        
        if row == 0:
            
            tam = sum(v[1] for v in list(values.values()))
            
            print('-'*(tam+1))
            print('|', end='')
            print(f'{title:^{tam-1}}', end='')
            print('|')
            print('-'*(tam+1))

        for label, (result, size) in values.items():
            size -= (len(label)+2)
            print(f'| {label} {result:>{size-2}} ', end='')

        print('|')