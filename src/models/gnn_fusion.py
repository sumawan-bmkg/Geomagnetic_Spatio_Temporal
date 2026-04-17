"""
Graph Neural Network (GNN) Fusion Layer

Implements spatial relationship learning between geomagnetic stations
using Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GraphConvLayer(nn.Module):
    """
    Graph Convolutional Layer for station feature aggregation.
    """
    
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.2):
        """
        Initialize Graph Convolutional Layer.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            dropout_rate: Dropout rate for regularization
        """
        super(GraphConvLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Linear transformation
        self.linear = nn.Linear(input_dim, output_dim)
        
        # Normalization and activation
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through graph convolution.
        
        Args:
            x: Node features (B, N, input_dim)
            edge_index: Edge indices (2, E)
            
        Returns:
            Updated node features (B, N, output_dim)
        """
        B, N, _ = x.shape
        
        # Linear transformation
        x_transformed = self.linear(x)  # (B, N, output_dim)
        
        # Message passing
        row, col = edge_index
        
        # Aggregate messages from neighbors
        x_aggregated = torch.zeros_like(x_transformed)
        
        for b in range(B):
            # For each edge, aggregate neighbor features
            for i in range(len(row)):
                src, dst = row[i], col[i]
                x_aggregated[b, dst] += x_transformed[b, src]
        
        # Normalize by degree (number of neighbors)
        degree = torch.zeros(N, device=x.device)
        for i in range(len(row)):
            degree[row[i]] += 1
        
        degree = degree.clamp(min=1)  # Avoid division by zero
        x_aggregated = x_aggregated / degree.view(1, -1, 1)
        
        # Add self-connection
        x_output = x_aggregated + x_transformed
        
        # Apply normalization and activation
        x_output = x_output.view(B * N, -1)
        x_output = self.batch_norm(x_output)
        x_output = self.activation(x_output)
        x_output = self.dropout(x_output)
        x_output = x_output.view(B, N, -1)
        
        return x_output


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer for adaptive spatial relationship learning.
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 num_heads: int = 4, dropout_rate: float = 0.2):
        """
        Initialize Graph Attention Layer.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            num_heads: Number of attention heads
            dropout_rate: Dropout rate
        """
        super(GraphAttentionLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        
        # Multi-head attention components
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        
        # Output projection
        self.output_proj = nn.Linear(output_dim, output_dim)
        
        # Normalization and regularization
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Attention scaling
        self.scale = (self.head_dim) ** -0.5
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through graph attention.
        
        Args:
            x: Node features (B, N, input_dim)
            edge_index: Edge indices (2, E)
            
        Returns:
            Updated node features (B, N, output_dim)
        """
        B, N, _ = x.shape
        
        # Compute queries, keys, values
        Q = self.query(x).view(B, N, self.num_heads, self.head_dim)  # (B, N, H, D)
        K = self.key(x).view(B, N, self.num_heads, self.head_dim)
        V = self.value(x).view(B, N, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (B, H, N, D)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        
        # Create attention mask based on graph edges
        attention_mask = self._create_attention_mask(edge_index, N, x.device)
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        
        # Apply mask (set non-connected nodes to -inf)
        attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)  # (B, H, N, D)
        
        # Concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous()  # (B, N, H, D)
        attended_values = attended_values.view(B, N, self.output_dim)  # (B, N, output_dim)
        
        # Output projection
        output = self.output_proj(attended_values)
        
        # Residual connection and layer normalization
        if self.input_dim == self.output_dim:
            output = self.layer_norm(output + x)
        else:
            output = self.layer_norm(output)
        
        return output
    
    def _create_attention_mask(self, edge_index: torch.Tensor, 
                             num_nodes: int, device: torch.device) -> torch.Tensor:
        """
        Create attention mask based on graph connectivity.
        
        Args:
            edge_index: Edge indices (2, E)
            num_nodes: Number of nodes
            device: Device for tensor creation
            
        Returns:
            Attention mask (N, N)
        """
        mask = torch.zeros(num_nodes, num_nodes, device=device)
        
        # Add self-connections
        mask.fill_diagonal_(1)
        
        # Add edges
        row, col = edge_index
        mask[row, col] = 1
        
        return mask


class StationSqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for station-level feature recalibration.
    Emphasizes stations with high anomaly variance.
    """
    
    def __init__(self, n_stations: int, reduction_ratio: int = 4):
        """
        Initialize Station SE block.
        
        Args:
            n_stations: Number of stations
            reduction_ratio: Reduction ratio for the bottleneck
        """
        super(StationSqueezeExcitation, self).__init__()
        
        self.n_stations = n_stations
        
        # Squeeze-and-Excitation MLP
        self.se_mlp = nn.Sequential(
            nn.Linear(n_stations, n_stations // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(n_stations // reduction_ratio, n_stations),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SE block.
        
        Args:
            x: Input features (B, S, D)
            
        Returns:
            Weighted features (B, S, D)
        """
        B, S, D = x.shape
        
        # Squeeze: Compute variance across feature dimension for each station
        # This captures "anomaly intensity" per station
        squeeze = torch.var(x, dim=2)  # (B, S)
        
        # Excitation: Compute station weights
        excitation = self.se_mlp(squeeze)  # (B, S)
        
        # Reshape and apply weights
        excitation = excitation.view(B, S, 1)
        
        return x * excitation


class GNNFusionLayer(nn.Module):
    """
    GNN Fusion Layer combining Graph Convolution and Graph Attention
    for learning spatial relationships between geomagnetic stations.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3,
                 n_stations: int = 8, dropout_rate: float = 0.2,
                 use_attention: bool = True, use_se: bool = True):
        """
        Initialize GNN Fusion Layer.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            n_stations: Number of stations
            dropout_rate: Dropout rate
            use_attention: Whether to use attention mechanism
            use_se: Whether to use Squeeze-and-Excitation block
        """
        super(GNNFusionLayer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_stations = n_stations
        self.use_attention = use_attention
        self.use_se = use_se
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        
        for i in range(num_layers):
            if use_attention and i > 0:  # Use attention in later layers
                layer = GraphAttentionLayer(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    num_heads=4,
                    dropout_rate=dropout_rate
                )
            else:
                layer = GraphConvLayer(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    dropout_rate=dropout_rate
                )
            
            self.gnn_layers.append(layer)
        
        # Station Squeeze-and-Excitation
        if use_se:
            self.station_se = StationSqueezeExcitation(n_stations)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Global pooling for station-level features
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        logger.info(f"GNNFusionLayer initialized:")
        logger.info(f"  Input dim: {input_dim}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Num layers: {num_layers}")
        logger.info(f"  Use attention: {use_attention}")
        logger.info(f"  Use SE: {use_se}")
    
    def forward(self, x: torch.Tensor, graph: object) -> torch.Tensor:
        """
        Forward pass through GNN fusion.
        
        Args:
            x: Station features (B, S, input_dim)
            graph: Graph object with edge_index
            
        Returns:
            Fused features (B, S, hidden_dim)
        """
        B, S, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)  # (B, S, hidden_dim)
        
        # Apply GNN layers
        for i, layer in enumerate(self.gnn_layers):
            x_residual = x
            x = layer(x, graph.edge_index)
            
            # Residual connection (if dimensions match)
            if x.shape == x_residual.shape:
                x = x + x_residual
        
        # Apply Station SE weighting
        if self.use_se:
            x = self.station_se(x)
        
        # Output projection
        x = self.output_proj(x)  # (B, S, hidden_dim)
        
        return x

    
    def get_attention_weights(self, x: torch.Tensor, graph: object) -> Optional[torch.Tensor]:
        """
        Get attention weights from the last attention layer.
        
        Args:
            x: Station features (B, S, input_dim)
            graph: Graph object with edge_index
            
        Returns:
            Attention weights if available
        """
        if not self.use_attention:
            return None
        
        # Find last attention layer
        attention_layer = None
        for layer in reversed(self.gnn_layers):
            if isinstance(layer, GraphAttentionLayer):
                attention_layer = layer
                break
        
        if attention_layer is None:
            return None
        
        # Forward pass to get attention weights
        x = self.input_proj(x)
        
        for layer in self.gnn_layers:
            if layer == attention_layer:
                # Extract attention weights (this would require modification of the attention layer)
                # For now, return None
                return None
            x = layer(x, graph.edge_index)
        
        return None


if __name__ == '__main__':
    # Test GNN Fusion Layer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create test data
    batch_size = 2
    n_stations = 8
    input_dim = 1280
    hidden_dim = 256
    
    x = torch.randn(batch_size, n_stations, input_dim).to(device)
    
    # Create simple graph (fully connected)
    edge_list = []
    for i in range(n_stations):
        for j in range(n_stations):
            if i != j:
                edge_list.append([i, j])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)
    
    class Graph:
        def __init__(self, edge_index):
            self.edge_index = edge_index
    
    graph = Graph(edge_index)
    
    # Create GNN layer
    gnn = GNNFusionLayer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=3,
        n_stations=n_stations,
        use_attention=True
    ).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = gnn(x, graph)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"GNN Fusion Layer test completed successfully!")