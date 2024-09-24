import torch
import torch.nn as nn
from torch_geometric.data import Batch

def train_model(model, loader, optimizer, device):
    '''
    
    '''

    model.train()

    total_loss = 0

    for data in loader:
        data = data.to(device)

        optimizer.zero_grad()

        pred = model(data)
        
        data_size = data.spectrum.shape[0] // 300
        data.spectrum = data.spectrum.view(data_size, 300)

        loss = nn.MSELoss()(pred, data.spectrum).float()

        total_loss += loss.item() / data.num_graphs

        loss.backward()

        optimizer.step()

        # out = pred[0].detach().cpu().numpy()
        # true = data.spectrum[0].detach().cpu().numpy()

    return total_loss #, embedding#, true, out

def train_schnet(model, loader, optimizer, device):
    '''
    '''

    model.train()

    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        z = data.z
        pos = data.x.float()
        pred = model(z, pos, data.batch)
        # data_size = data.spectrum.shape[0] // 200
        # data.spectrum = data.spectrum.view(data_size, 200)
        loss = nn.MSELoss()(pred, data.spectrum).float()
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()

        return total_loss / len(loader.dataset)

def val_test(model, loader, device):
    '''
    
    '''

    model.eval()

    total_loss = 0

    for data in loader:
        data = data.to(device)

        pred = model(data)

        data_size = data.spectrum.shape[0] // 200
        data.spectrum = data.spectrum.view(data_size, 200)
        
        loss = nn.MSELoss()(pred, data.spectrum)

        total_loss += loss.item() * data.num_graphs

        out = pred[0].detach().cpu().numpy()
        true = data.spectrum[0].detach().cpu().numpy()

    return total_loss / len(loader.dataset)#, out, true

def val_schnet(model, loader, device):
    '''
    '''

    model.eval()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        z = data.z
        pos = data.x.float()
        pred = model(z, pos, data.batch)
        # data_size = data.spectrum.shape[0] // 200
        # data.spectrum = data.spectrum.view(data_size, 200)
        loss = nn.MSELoss()(pred, data.spectrum)
        total_loss += loss.item() * data.num_graphs
        return total_loss / len(loader.dataset)


def get_spec_prediction(model, index, dataset, device):
    '''
    
    '''
    # --- Set the model to evaluation mode
    model.eval()

    # --- Get a single graph from the test dataset
    graph_index = index
    graph_data = dataset[graph_index].to(device)
    data = Batch.from_data_list([graph_data])

    # --- Pass the graph through the model
    with torch.no_grad():
        pred = model(data)

    # ---
    true_spectrum = graph_data.spectrum.cpu().numpy()
    predicted_spectrum = pred.cpu().numpy()
    predicted_spectrum = predicted_spectrum.reshape(-1)

    return predicted_spectrum, true_spectrum
