from typing import Union
import exputils as eu
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import exputils.data.logging as log
import gc

class SKLFit:

    @staticmethod
    def default_config():
        def_config = eu.AttrDict()
        def_config.device = "cpu"
        def_config.n_epochs = 100

        return def_config
    

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = eu.combine_dicts(kwargs, config, self.default_config())


    def train(self,model,train_dataset:Dataset,val_dataset:Union[None,Dataset] = None,test_dataset:Union[None,Dataset] = None,logger=None):

        if logger == None:
            logger = log

        criterion = torch.nn.MSELoss()

        x,y = train_dataset[:]
        x_test,y_test = test_dataset[:]
        x_val,y_val = val_dataset[:]

        model.fit(x,y)

        print(f"X: {x.shape}")
        y_pred = model(x)
        
        print(f"Y pred: {y_pred.shape}")
        loss = criterion(y_pred,y)
        for epoch in range(self.config.n_epochs):
            logger.add_value('train_loss',loss)
            logger.add_value('epoch', epoch)


        #del y_pred
        #gc.collect()

        y_pred = model(x_val)
        loss = criterion(y_pred,y_val)
        for epoch in range(self.config.n_epochs):
            logger.add_value('val_loss',loss)

        y_pred = model(x_test)
        loss = criterion(y_pred,y_test)
        for epoch in range(self.config.n_epochs):
            logger.add_value('test_loss',loss)

        logger.save()

        return model