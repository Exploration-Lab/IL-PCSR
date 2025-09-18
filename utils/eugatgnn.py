from utils.EUGATConv import EUGATConv
class EUGATGNN(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, dropout, num_head):
        super(EUGATGNN, self).__init__()
        self.hidden_size = h_dim
        self.in_dim = in_dim
        self.EUGATConv1 = EUGATConv(in_feats=in_dim, edge_feats=in_dim, out_feats=out_dim, out_edge_feats=out_dim, num_heads=num_head,allow_zero_in_degree=True)
        self.EUGATConv2 = EUGATConv(in_feats=in_dim, edge_feats=in_dim, out_feats=out_dim, out_edge_feats=out_dim, num_heads=num_head,allow_zero_in_degree=True)
        self.embedding_dropout1 = nn.Dropout(dropout)
        self.embedding_dropout2 = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        if self.hidden_size == 0:
            stdv = 1.0 / math.sqrt(self.in_dim)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)
        else:
            stdv = 1.0 / math.sqrt(self.hidden_size)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)

    def forward(self, g, node_feats, edge_feats):
        ##Layer 1
        h = self.EUGATConv1(g, node_feats, edge_feats) 
        h_0 = th.squeeze(h[0]) ##h_0: node feature, h_1: edge feature
        h_1 = th.squeeze(h[1])
        h_0 = self.embedding_dropout1(h_0)
        h_1 = self.embedding_dropout2(h_1)
        h_0 = F.relu(h_0)+node_feats
        h_1 = F.relu(h_1)+edge_feats
        ##Layer2
        h = self.EUGATConv2(g, h_0, h_1) 
        h_0 = F.relu(h_0)
        h = th.squeeze(h[0])+node_feats
        #h = h[g.ndata["node_mask"].bool()]
        return h
    
