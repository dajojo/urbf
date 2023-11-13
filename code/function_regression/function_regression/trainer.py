from typing import Union
import exputils as eu
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import exputils.data.logging as log

class Trainer:

    @staticmethod
    def default_config():
        def_config = eu.AttrDict()

        def_config.learning_rate = 0.01
        def_config.n_epochs = 100
        def_config.batch_size = 64

        return def_config
    

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = eu.combine_dicts(kwargs, config, self.default_config())


    def train(self,model,train_dataset:Dataset,test_dataset:Union[None,Dataset] = None):
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

        if test_dataset != None:
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=True)
        else:
            test_loader = None

        # Define an optimizer and a loss function
        optimizer = torch.optim.SGD(list(model.parameters()), lr=0.01)
        criterion = torch.nn.MSELoss()

        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        eu.misc.update_status(f'Epoch 0/{self.config.n_epochs} - Loss: -')


        for epoch in range(self.config.n_epochs):
            model.train()  # Set the model to training mode
            running_train_loss = 0.0
            
            for i, (inputs, labels) in enumerate(train_loader):

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass: compute the model output
                outputs = model(inputs)#.squeeze()

                # Compute the loss
                loss = criterion(outputs, labels)

                # Backward pass: compute the gradient of the loss with respect to model parameters
                loss.backward()

                # Perform a single optimization step (parameter update)
                optimizer.step()

                # Accumulate the running loss
                running_train_loss += loss.item()#.detach().numpy()

            # Print statistics after every epoch
            log.add_value('train_loss',running_train_loss)
            log.add_value('means',model.layers[0].means)
            

            if test_loader != None:
                model.eval()
                running_test_loss = 0.0
                for i, (inputs, labels) in enumerate(test_loader):
                    # Forward pass: compute the model output
                    outputs = model(inputs)#.squeeze()
                    # Compute the loss
                    loss = criterion(outputs, labels)
                    # Accumulate the running loss
                    running_test_loss += loss.item()
                log.add_value('test_loss',running_test_loss)
            log.add_value('epoch', epoch)

            eu.misc.update_status(f'Epoch {epoch + 1}/{self.config.n_epochs} - Train Loss: {running_train_loss/len(train_loader):.4f}')


        log.save()

        return model