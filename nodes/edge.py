class EdgeServer:
    def __init__(self, id, assigned_clients, input_dim):
        self.id = id
        self.clients = assigned_clients
        self.param_dim = input_dim