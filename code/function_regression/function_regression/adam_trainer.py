from typing import Union
import exputils as eu
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import exputils.data.logging as log

class AdamTrainer:

    @staticmethod
    def default_config():
        def_config = eu.AttrDict()
        def_config.learning_rate = 0.01
        def_config.urbf_learning_rate = None
        def_config.n_epochs = 100
        def_config.batch_size = 32
        def_config.device = "cuda"
        def_config.use_adaptive_range = False
        return def_config
    

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = eu.combine_dicts(kwargs, config, self.default_config())


    def train(self,model,train_dataset:Dataset,test_dataset:Union[None,Dataset] = None,logger=None):

        if logger == None:
            logger = log


        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

        if test_dataset != None:
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=True)
        else:
            test_loader = None

        if self.config.urbf_learning_rate == None:
            self.config.urbf_learning_rate = self.config.learning_rate



        ## Minor experiment checking the influence of different learning rates...
        # Define an optimizer and a loss function
        optimizer = torch.optim.Adam([{'params':model.params.mlp.parameters()},
                                     {'params':model.params.urbf.parameters(), 'lr': self.config.urbf_learning_rate,'weight_decay': 0}], lr=self.config.learning_rate)
        
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
                #print(f"Updated mean: {means.shape}")

                for idx,mean in enumerate(means):
                    logger.add_value(f"mean{idx}",mean)

                vars = model.layers[0].rbf_layer.state_dict()["vars"].cpu().detach().numpy()
                #print(f"Updated vars: {vars.shape}")

                for idx,var in enumerate(vars):
                    logger.add_value(f"var{idx}",var)
                
                if hasattr(model.layers[0].rbf_layer,'coefs'):
                    coefs = model.layers[0].rbf_layer.coefs.cpu().detach().numpy()
                    #print(f"Updated coefs: {coefs.shape}")

                    for idx,coef in enumerate(coefs):
                        logger.add_value(f"coef{idx}",coef)

                if hasattr(model.layers[0],'expansion_mapping'):
                    expansion_mapping = model.layers[0].expansion_mapping.cpu().detach().numpy()
                    #print(f"Updated expansion_mapping: {expansion_mapping.shape}")

                    for idx,expansion_map in enumerate(expansion_mapping):
                        logger.add_value(f"expansion_map{idx}",expansion_map.argmax()+expansion_map.sum())

                if hasattr(model.layers[0],'significance'):
                    significances = model.layers[0].significance.cpu().detach().numpy()
                    #print(f"Updated significance: {significances.shape}")

                    for idx,significance in enumerate(significances):
                        logger.add_value(f"significance{idx}",significance)

                if hasattr(model.layers[0],'linear_layer_grad_output') and model.layers[0].linear_layer_grad_output != None:
                    linear_layer_grad_outputs = model.layers[0].linear_layer_grad_output[0]#.mean(dim=0)
                    #print(f"Updated linear_layer_grad_output: {linear_layer_grad_outputs.shape}")

                    for idx,linear_layer_grad_output in enumerate(linear_layer_grad_outputs):
                        logger.add_value(f"linear_layer_grad_output{idx}",linear_layer_grad_output.cpu().detach().numpy())


                if hasattr(model.layers[0],'input') and model.layers[0].input != None:
                    inputs = model.layers[0].input[0]#.mean(dim=0)
                    #print(f"Updated input: {inputs.shape}")

                    for idx,input in enumerate(inputs):
                        logger.add_value(f"input{idx}",input.cpu().detach().numpy())



            for i, (inputs, labels) in enumerate(train_loader):

                ### TEMPORARY FOR TESTING ONLY ###
                #inputs = inputs[:10+i]
                #labels = labels[:10+i]


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
                logger.add_value('train_loss_fine',loss.item())


            # Print statistics after every epoch
            logger.add_value('train_loss',running_train_loss)

            # Keep track of the learning rate... 
            logger.add_value('lr',optimizer.param_groups[0]['lr'])
            logger.add_value('urbf_lr',optimizer.param_groups[1]['lr'])
            

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
                logger.add_value('test_loss',running_test_loss)
            
            logger.add_value('epoch', epoch)

        logger.save()

        return model