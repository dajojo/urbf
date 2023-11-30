from typing import Union
import exputils as eu
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import exputils.data.logging as log

class SGDTrainer:

    @staticmethod
    def default_config():
        def_config = eu.AttrDict()
        def_config.learning_rate = 0.01
        def_config.urbf_learning_rate = None
        def_config.n_epochs = 100
        def_config.batch_size = 32
        def_config.device = "cpu"

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

        if self.config.urbf_learning_rate == None:
            self.config.urbf_learning_rate = self.config.learning_rate



        ## Minor experiment checking the influence of different learning rates...
        # Define an optimizer and a loss function
        optimizer = torch.optim.SGD([{'params':model.params.mlp.parameters()},
                                     {'params':model.params.urbf.parameters(), 'lr': self.config.urbf_learning_rate}], lr=self.config.learning_rate)
        
        # Define an optimizer and a loss function
        #optimizer = torch.optim.SGD([model.parameters()], lr=self.config.learning_rate)
        
        criterion = torch.nn.MSELoss()

        # Move model to GPU if available
        device = self.config.device #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        eu.misc.update_status(f'Epoch 0/{self.config.n_epochs} - Loss: -')


        for epoch in range(self.config.n_epochs):
            model.train()  # Set the model to training mode
            running_train_loss = 0.0
            
            if hasattr(model.layers[0],'rbf_layer'):
                means = model.layers[0].rbf_layer.state_dict()["means"].cpu().detach().numpy()
                print(f"Updated mean: {means}")

                for idx,mean in enumerate(means):
                    log.add_value(f"mean{idx}",mean)

                vars = model.layers[0].rbf_layer.state_dict()["vars"].cpu().detach().numpy()
                print(f"Updated vars: {vars}")

                for idx,var in enumerate(vars):
                    log.add_value(f"var{idx}",var)

                coefs = model.layers[0].rbf_layer.state_dict()["coefs"].cpu().detach().numpy()
                print(f"Updated coefs: {coefs}")

                for idx,coef in enumerate(coefs):
                    log.add_value(f"coef{idx}",coef)



            for i, (inputs, labels) in enumerate(train_loader):

                # Zero the parameter gradients
                optimizer.zero_grad()

                #inputs.require_grad = True
                # Forward pass: compute the model output
                outputs = model(inputs.to(device))#.squeeze()

                # Compute the loss
                loss = criterion(outputs, labels.to(device))

                # Backward pass: compute the gradient of the loss with respect to model parameters
                loss.backward()

                # Perform a single optimization step (parameter update)
                optimizer.step()

                # Accumulate the running loss
                running_train_loss += loss.item()#.detach().numpy()
                log.add_value('train_loss_fine',loss.item())


            # Print statistics after every epoch
            log.add_value('train_loss',running_train_loss)

            eu.misc.update_status(f'Epoch {epoch + 1}/{self.config.n_epochs} - Train Loss: {running_train_loss/len(train_loader):.4f}')
        

            if test_loader != None:
                model.eval()
                running_test_loss = 0.0
                for i, (inputs, labels) in enumerate(test_loader):
                    # Forward pass: compute the model output
                    outputs = model(inputs.to(device))#.squeeze()
                    # Compute the loss
                    loss = criterion(outputs, labels.to(device))
                    # Accumulate the running loss
                    running_test_loss += loss.item()
                log.add_value('test_loss',running_test_loss)
            log.add_value('epoch', epoch)


        log.save()

        return model