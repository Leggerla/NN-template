class model(nn.Module):

  def __init__(self, 
               input_dim = ,
               hidden_dim = ,
               output_dim = ,
               dropout_rate = ):
    super().__init__()

    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.dropout_rate = dropout_rate

    self.dp = nn.Dropout(self.dropout_rate)

  def forward(self, x):

    return x
