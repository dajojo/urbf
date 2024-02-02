from typing import Union
import exputils as eu
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import exputils.data.logging as log
from function_regression.models.rbf_mlp import RBFMLP


class AdamTrainer:

    @staticmethod
    def default_config():
        def_config = eu.AttrDict()
        def_config.learning_rate = 0.01
        def_config.urbf_learning_rate = None
        def_config.rbf_learning_rate = None
        def_config.n_epochs = 100
        def_config.batch_size = 32
        def_config.device = "cpu"
        def_config.is_classification = False
        return def_config
    

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = eu.combine_dicts(kwargs, config, self.default_config())


    def train(self,model,train_dataset:Dataset,val_dataset:Union[None,Dataset] = None,test_dataset:Union[None,Dataset] = None,logger=None):

        if logger == None:
            logger = log


        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

        if val_dataset != None:
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=True)
        else:
            val_loader = None

        if test_dataset != None:
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=True)
        else:
            test_loader = None

        if self.config.urbf_learning_rate == None:
            self.config.urbf_learning_rate = self.config.learning_rate

        if self.config.rbf_learning_rate == None:
            self.config.rbf_learning_rate = self.config.learning_rate

        if hasattr(model.params,"urbf_linear") or hasattr(model.params,"urbf"):

            # Define an optimizer and a loss function
            optimizer = torch.optim.Adam([{'params':model.params.mlp.parameters()},
                                        {'params':model.params.urbf_linear.parameters(),'lr': self.config.learning_rate,'weight_decay': 0.0001},
                                        {'params':model.params.urbf.parameters(), 'lr': self.config.urbf_learning_rate,'weight_decay': 0},
                      ], lr=self.config.learning_rate,weight_decay=0)
            
        elif isinstance(model, RBFMLP):
            # Define an optimizer and a loss function
            optimizer = torch.optim.Adam([{'params':model.params.mlp.parameters()},
                                        {'params':model.params.rbf_linear.parameters(),'lr': self.config.learning_rate,'weight_decay': 0.0001},
                                        {'params':model.params.rbf.parameters(), 'lr': self.config.rbf_learning_rate,'weight_decay': 0},], lr=self.config.learning_rate,weight_decay=0)
            
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate,weight_decay=0)


        if self.config.is_classification:
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.MSELoss()

        device = self.config.device

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            device = "mps" if torch.backends.mps.is_built() else device



        model.to(device)

        eu.misc.update_status(f'Epoch 0/{self.config.n_epochs} - Loss: -')


        for epoch in range(self.config.n_epochs):
            model.train()  # Set the model to training mode
            running_train_loss = 0.0
            
            if hasattr(model.layers[0],'rbf_layer'):
                means = model.layers[0].rbf_layer.state_dict()["means"].cpu().detach().numpy()
                for idx,mean in enumerate(means):
                    logger.add_value(f"mean{idx}",mean)

                stds = model.layers[0].rbf_layer.state_dict()["stds"].cpu().detach().numpy()
                for idx,std in enumerate(stds):
                    logger.add_value(f"std{idx}",std)
                

                if hasattr(model.layers[0],'expansion_mapping'):
                    expansion_mapping = model.layers[0].expansion_mapping.cpu().detach().numpy()
                    for idx,expansion_map in enumerate(expansion_mapping):
                        logger.add_value(f"expansion_map{idx}",expansion_map.argmax()+expansion_map.sum())

            for i, (inputs, labels) in enumerate(train_loader):

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass: compute the model output
                outputs = model(inputs.to(device))#.squeeze()

                if self.config.is_classification and len(outputs.shape) > 1:
                    labels = labels.squeeze().long()
                # Compute the loss
                loss = criterion(outputs, labels.to(device))


                # Backward pass: compute the gradient of the loss with respect to model parameters
                loss.backward()

                # Perform a single optimization step (parameter update)
                optimizer.step()

                # Accumulate the running loss
                running_train_loss += loss.item()#.detach().numpy()
                #logger.add_value('train_loss_fine',loss.item())

            # Print statistics after every epoch
            logger.add_value('train_loss',running_train_loss)
            
            print(f'Epoch {epoch + 1}/{self.config.n_epochs} - Train Loss: {running_train_loss/len(train_loader):.4f}')
            eu.misc.update_status(f'Epoch {epoch + 1}/{self.config.n_epochs} - Train Loss: {running_train_loss/len(train_loader):.4f}')
        
            if val_loader != None:
                model.eval()
                running_val_loss = 0.0
                for i, (inputs, labels) in enumerate(val_loader):
                    # Forward pass: compute the model output
                    outputs = model(inputs.to(device))#.squeeze()
                    # Compute the loss
                    if self.config.is_classification and len(outputs.shape) > 1:
                        labels = labels.squeeze().long()
                    
                    loss = criterion(outputs, labels.to(device))
                    # Accumulate the running loss
                    running_val_loss += loss.item()
                logger.add_value('val_loss',running_val_loss)


            if test_loader != None:
                model.eval()
                running_test_loss = 0.0
                for i, (inputs, labels) in enumerate(test_loader):
                    # Forward pass: compute the model output
                    outputs = model(inputs.to(device))#.squeeze()
                    # Compute the loss

                    if self.config.is_classification and len(outputs.shape) > 1:
                        labels = labels.squeeze().long()

                    loss = criterion(outputs, labels.to(device))
                    # Accumulate the running loss
                    running_test_loss += loss.item()
                logger.add_value('test_loss',running_test_loss)
            
            logger.add_value('epoch', epoch)

        logger.save()

        return model