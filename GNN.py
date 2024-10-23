import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv, GINConv, GINEConv, GATv2Conv, MLP, SAGEConv
from torch_geometric.nn import aggr
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning as L

class QM9_GNN(L.LightningModule):
    '''
        Graph neural network class to run various types of GNNs
    '''
    def __init__(self, num_tasks, num_layer=4, in_channels=[33,100,100,100], out_channels=[100,100,100,100],
                 gnn_type='gcn', heads=None, drop_ratio=0.5, graph_pooling="sum"):
        '''
            Args:
                num_tasks (int): number of labels to be predicted
                num_layer (int): number of layers in the NN
                in_channels (list): size of each input layer
                out_channels (list): size of each output layer
                gnn_type (str): the specific GNN to use
                heads (int): number of heads
                drop_ratio (float): the drop_ratio 
                graph_pooling (str): the pooling function for the GNN
        '''

        super(QM9_GNN, self).__init__()

        self.num_layer = num_layer
        self.gnn_type = gnn_type
        self.drop_ratio = drop_ratio
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self._initialize_weights()
        self.save_hyperparameters()

        # --- Sanity check number of layers
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        # --- Choose the selected pooling function
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # --- Create GNN model
        for i, in_c, out_c in zip(range(num_layer), in_channels, out_channels):
            if gnn_type == 'gin':
                mlp = MLP([in_c, in_c, out_c])
                self.convs.append(GINConv(nn=mlp, train_eps=False))
            if gnn_type == 'gine':
                mlp = MLP([in_c, out_c, out_c])
                self.convs.append(GINEConv(nn=mlp, eps=0.5, train_eps=True, edge_dim=5))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(in_c, out_c))
            elif gnn_type == 'gat':
                if i == 1:
                    self.convs.append(GATv2Conv(int(in_c), out_c, heads=int(heads), edge_dim=5))
                else:
                    self.convs.append(GATv2Conv(int(in_c*heads), out_c, heads=int(heads), edge_dim=5))
            elif gnn_type == 'sage':
                self.convs.append(SAGEConv(in_c, out_c, aggr=aggr.SumAggregation()))
            # elif gnn_type='mpnn':
            #     nn = Sequential(Linear(in_c, in_c), ReLU(), Linear(in_c, out_c * out_c))
            #     self.convs.append (NNConv(in_c, in_c, nn))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(out_c))

        # --- Add a final linear layer after GNN
        self.graph_pred_linear = torch.nn.Linear(self.out_channels[-1], self.num_tasks)      	


    def forward(self, batched_data):

        x = batched_data.x.float()
        edge_index = batched_data.edge_index
        edge_attr = batched_data.edge_attr.float()
        batch = batched_data.batch

        # --- Create list of features
        h_list = [x]

        # --- Pass through the GNN model
        for layer in range(self.num_layer):
            # --- Use edge_attr for required models
            if self.gnn_type == 'gine' or self.gnn_type == 'gat':
                h = self.convs[layer](h_list[layer], edge_index, edge_attr=edge_attr)
            else:
                h = self.convs[layer](h_list[layer], edge_index)

            h = self.batch_norms[layer](h)

            # --- Apply dropout to model
            if layer == self.num_layer - 1:
                # --- Remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            h_list.append(h)
             
        node_representation = h_list[-1]

        # --- Create a total graph embeding by pooling all nodes
        graph_embedding = self.pool(node_representation, batched_data.batch)
        # --- Activation function
        p = torch.nn.LeakyReLU(0.1)
        # --- Forward pass through linear layer
        out = p(self.graph_pred_linear(graph_embedding))

        return out#, node_embedding
    
    def training_step(self, train_batch, batch_idx):
        data = train_batch
        batch_size = len(data)
        spec_hat = self(data)
        loss = F.mse_loss(spec_hat.view(-1, 1), data.spectrum.view(-1, 1))
        self.log('train_loss', loss, logger=True, on_epoch=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        data = val_batch
        batch_size = len(data)
        spec_hat = self(data)
        loss = F.mse_loss(spec_hat.view(-1, 1), data.spectrum.view(-1, 1))
        self.log('val_loss', loss, logger=True, on_epoch=True, batch_size=batch_size)
        return loss
    
    def test_step(self, test_batch, batch_idx):
        data = test_batch
        spec_hat = self(data)
        loss = F.mse_loss(spec_hat.view(-1, 1), data.spectrum.view(-1, 1))
        self.log('test_loss', loss)
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(),
                         lr=0.01,
                         betas=(0.9, 0.999), 
                         eps=1e-08, 
                         amsgrad=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer, 
                    mode='min',
                    factor=0.5,
                    patience=100,
                    min_lr=0.0000001
                ),
                "monitor": "val_loss",
                "frequency": 1
            }
        }

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


