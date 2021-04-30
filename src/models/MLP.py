class MLP(nn.Module):
    """Create a Multilayer Perceptron Architecture.

        Args:
            hidden_dims (list): List with the number of neurons for each hidden layer
            dropout (float): Dropout rate
            function (function): Activation function used between layers
            out_size (int): number of classes"""
    def __init__(self, hidden_dims=[784,532,282,60,32], function=F.selu, dropout=0.2, out_size=15):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(152, hidden_dims[0])])
        self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_dims[0])])
        
        for idx in range(1,len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[idx-1], hidden_dims[idx]))
            self.bn.append(nn.BatchNorm1d(hidden_dims[idx]))
    
        self.out_layer = nn.Linear(hidden_dims[-1],out_size)
        self.dropout = nn.Dropout(dropout)
        self.F = function
        
    def forward(self,x):
        
        if self.training:
            
            for idx in range(len(self.layers)):
                x = self.layers[idx](x)
                x = self.F(x)
                x = self.dropout(x)
                x = self.bn[idx](x)
                
        else:
            for idx in range(len(self.layers)):
                x = self.layers[idx](x)
                x = self.F(x)
                x = self.bn[idx](x)

        y = self.out_layer(x)
        return {'logits': y, 'representation': x}
