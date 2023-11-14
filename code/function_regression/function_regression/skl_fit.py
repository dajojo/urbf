from typing import Union
import exputils as eu
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import exputils.data.logging as log

class SKLFit:

    @staticmethod
    def default_config():
        def_config = eu.AttrDict()
        def_config.device = "cpu"

        return def_config
    

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = eu.combine_dicts(kwargs, config, self.default_config())


    def train(self,model,train_dataset:Dataset,test_dataset:Union[None,Dataset] = None):
        criterion = torch.nn.MSELoss()

        x,y = train_dataset[:]
        x_test,y_test = test_dataset[:]

        model.fit(x,y)


        print(f"X: {x.shape}")
        y_pred = model(x)
        print(f"Y pred: {y_pred.shape}")
        loss = criterion(y_pred,y)

        log.add_value('train_loss',loss)

        y_pred = model(x_test)
        loss = criterion(y_pred,y_test)

        log.add_value('test_loss',loss)

        log.save()

        return model