"""
Utility functions for model construction and graph building.
"""
import torch
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from scipy.spatial.distance import pdist, squareform
import logging

logger = logging.getLogger(__name__)


def load_station_coordinates(csv_path: str) -> np.ndarray:
    """
    Load station coordinates from CSV file.
    
    Args:
        csv_path: Path to station coordinates CSV
        
    Returns:
        Array of coordinates (N, 2) - [latitude, longitude]
    """
    try:
        # Try different separators
        try:
            df = pd.read_csv(csv_path, sep=';')
        except:
            df = pd.read_csv(csv_path, sep=',')
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Remove empty rows
        df = df.dropna(subset=['Kode Stasiun'])
        df = df[df['Kode Stasiun'].str.strip() != '']
        
        # Extract coordinates
        coordinates = df[['Latitude', 'Longitude']].values
        
        logger.info(f"Loaded {len(coordinates)} station coordinates from {csv_path}")
        
        return coordinates
        
    except Exception as e:
        logger.error(f"Error loading station coordinates: {e}")
        return None


def calculate_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    """
    Calculate distance matrix between stations using Haversine formula.
    
    Args:
        coordinates: Station coordinates (N, 2) - [lat, lon] in degrees
        
    Returns:
        Distance matrix (N, N) in kilometers
    """
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate Haversine distance between two points."""
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in kilometers
        R = 6371.0
        
        return R * c
    
    n_stations = len(coordinates)
    distance_matrix = np.zeros((n_stations, n_stations))
    
    for i in range(n_stations):
        for j in range(n_stations):
            if i != j:
                lat1, lon1 = coordinates[i]
                lat2, lon2 = coordinates[j]
                distance_matrix[i, j] = haversine_distance(lat1, lon1, lat2, lon2)
    
    return distance_matrix


def build_station_graph(coordinates: np.ndarray, 
                       connection_type: str = 'knn',
                       k: int = 3,
                       distance_threshold: float = 500.0) -> object:
    """
    Build graph connectivity between stations based on coordinates.
    
    Args:
        coordinates: Station coordinates (N, 2) - [lat, lon]
        connection_type: Type of connectivity ('knn', 'threshold', 'fully_connected')
        k: Number of nearest neighbors for KNN
        distance_threshold: Distance threshold in km for threshold connectivity
        
    Returns:
        Graph object with edge_index tensor
    """
    n_stations = len(coordinates)
    
    if connection_type == 'fully_connected':
        # Fully connected graph
        edge_list = []
        for i in range(n_stations):
            for j in range(n_stations):
                if i != j:
                    edge_list.append([i, j])
    
    elif connection_type == 'knn':
        # K-nearest neighbors graph
        distance_matrix = calculate_distance_matrix(coordinates)
        
        edge_list = []
        for i in range(n_stations):
            # Find k nearest neighbors (excluding self)
            distances = distance_matrix[i]
            nearest_indices = np.argsort(distances)[1:k+1]  # Exclude self (index 0)
            
            for j in nearest_indices:
                edge_list.append([i, j])
    
    elif connection_type == 'threshold':
        # Distance threshold graph
        distance_matrix = calculate_distance_matrix(coordinates)
        
        edge_list = []
        for i in range(n_stations):
            for j in range(n_stations):
                if i != j and distance_matrix[i, j] <= distance_threshold:
                    edge_list.append([i, j])
    
    else:
        raise ValueError(f"Unknown connection type: {connection_type}")
    
    # Convert to tensor
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        # Fallback to self-loops if no edges
        edge_index = torch.tensor([[i, i] for i in range(n_stations)], 
                                 dtype=torch.long).t().contiguous()
    
    # Create graph object
    class StationGraph:
        def __init__(self, edge_index, coordinates, distance_matrix=None):
            self.edge_index = edge_index
            self.coordinates = coordinates
            self.distance_matrix = distance_matrix
            self.n_nodes = len(coordinates)
            self.n_edges = edge_index.shape[1]
    
    distance_matrix = calculate_distance_matrix(coordinates)
    graph = StationGraph(edge_index, coordinates, distance_matrix)
    
    logger.info(f"Built station graph:")
    logger.info(f"  Nodes: {graph.n_nodes}")
    logger.info(f"  Edges: {graph.n_edges}")
    logger.info(f"  Connection type: {connection_type}")
    
    return graph


def create_magnitude_class_mapping(magnitude_ranges: List[Tuple[float, float]] = None) -> dict:
    """
    Create magnitude class mapping for classification.
    
    Args:
        magnitude_ranges: List of (min, max) magnitude ranges
        
    Returns:
        Dictionary mapping class indices to magnitude ranges
    """
    if magnitude_ranges is None:
        # Default magnitude classes
        magnitude_ranges = [
            (0.0, 4.0),   # Class 0: Very small
            (4.0, 5.0),   # Class 1: Small
            (5.0, 6.0),   # Class 2: Moderate
            (6.0, 7.0),   # Class 3: Strong
            (7.0, 10.0)   # Class 4: Major/Great
        ]
    
    class_mapping = {}
    for i, (min_mag, max_mag) in enumerate(magnitude_ranges):
        class_mapping[i] = {
            'range': (min_mag, max_mag),
            'label': f'M{min_mag:.1f}-{max_mag:.1f}'
        }
    
    return class_mapping


def magnitude_to_class(magnitude: float, class_mapping: dict = None) -> int:
    """
    Convert continuous magnitude to class index.
    
    Args:
        magnitude: Continuous magnitude value
        class_mapping: Magnitude class mapping
        
    Returns:
        Class index
    """
    if class_mapping is None:
        class_mapping = create_magnitude_class_mapping()
    
    for class_idx, class_info in class_mapping.items():
        min_mag, max_mag = class_info['range']
        if min_mag <= magnitude < max_mag:
            return class_idx
    
    # Default to highest class if magnitude exceeds all ranges
    return len(class_mapping) - 1


def class_to_magnitude_range(class_idx: int, class_mapping: dict = None) -> Tuple[float, float]:
    """
    Convert class index to magnitude range.
    
    Args:
        class_idx: Class index
        class_mapping: Magnitude class mapping
        
    Returns:
        Magnitude range (min, max)
    """
    if class_mapping is None:
        class_mapping = create_magnitude_class_mapping()
    
    if class_idx in class_mapping:
        return class_mapping[class_idx]['range']
    else:
        raise ValueError(f"Invalid class index: {class_idx}")


def normalize_coordinates(coordinates: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Normalize coordinates to [0, 1] range.
    
    Args:
        coordinates: Raw coordinates (N, 2)
        
    Returns:
        Normalized coordinates and normalization parameters
    """
    min_coords = coordinates.min(axis=0)
    max_coords = coordinates.max(axis=0)
    
    normalized = (coordinates - min_coords) / (max_coords - min_coords)
    
    normalization_params = {
        'min_coords': min_coords,
        'max_coords': max_coords
    }
    
    return normalized, normalization_params


def denormalize_coordinates(normalized_coords: np.ndarray, 
                          normalization_params: dict) -> np.ndarray:
    """
    Denormalize coordinates back to original scale.
    
    Args:
        normalized_coords: Normalized coordinates
        normalization_params: Normalization parameters
        
    Returns:
        Original scale coordinates
    """
    min_coords = normalization_params['min_coords']
    max_coords = normalization_params['max_coords']
    
    return normalized_coords * (max_coords - min_coords) + min_coords


def calculate_graph_statistics(graph: object) -> dict:
    """
    Calculate graph statistics for analysis.
    
    Args:
        graph: Graph object with edge_index
        
    Returns:
        Dictionary of graph statistics
    """
    edge_index = graph.edge_index
    n_nodes = graph.n_nodes
    n_edges = graph.n_edges
    
    # Calculate node degrees
    degrees = torch.zeros(n_nodes, dtype=torch.long)
    for i in range(n_edges):
        src = edge_index[0, i]
        degrees[src] += 1
    
    # Calculate statistics
    stats = {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'avg_degree': degrees.float().mean().item(),
        'max_degree': degrees.max().item(),
        'min_degree': degrees.min().item(),
        'density': n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
    }
    
    return stats


def visualize_station_graph(graph: object, save_path: str = None) -> None:
    """
    Visualize station graph connectivity.
    
    Args:
        graph: Graph object with coordinates and edge_index
        save_path: Path to save visualization
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes with positions
        coordinates = graph.coordinates
        for i in range(len(coordinates)):
            G.add_node(i, pos=(coordinates[i, 1], coordinates[i, 0]))  # (lon, lat)
        
        # Add edges
        edge_index = graph.edge_index
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            G.add_edge(src.item(), dst.item())
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Get positions
        pos = nx.get_node_attributes(G, 'pos')
        
        # Draw graph
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=10, font_weight='bold')
        
        plt.title('Station Graph Connectivity')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Graph visualization saved to {save_path}")
        
        plt.show()
        
    except ImportError:
        logger.warning("NetworkX or matplotlib not available for visualization")


if __name__ == '__main__':
    # Test utility functions
    print("Testing utility functions...")
    
    # Test coordinate loading (with dummy data)
    dummy_coordinates = np.array([
        [-5.0, 100.0],  # Station 1
        [-6.0, 101.0],  # Station 2
        [-7.0, 102.0],  # Station 3
        [-8.0, 103.0],  # Station 4
    ])
    
    print(f"Dummy coordinates shape: {dummy_coordinates.shape}")
    
    # Test distance matrix calculation
    distance_matrix = calculate_distance_matrix(dummy_coordinates)
    print(f"Distance matrix shape: {distance_matrix.shape}")
    print(f"Sample distances: {distance_matrix[0, 1:]}")
    
    # Test graph building
    for connection_type in ['fully_connected', 'knn', 'threshold']:
        graph = build_station_graph(dummy_coordinates, connection_type=connection_type)
        stats = calculate_graph_statistics(graph)
        print(f"\n{connection_type} graph stats: {stats}")
    
    # Test magnitude class mapping
    class_mapping = create_magnitude_class_mapping()
    print(f"\nMagnitude class mapping:")
    for class_idx, info in class_mapping.items():
        print(f"  Class {class_idx}: {info['label']} {info['range']}")
    
    # Test magnitude conversion
    test_magnitudes = [3.5, 4.5, 5.5, 6.5, 7.5]
    for mag in test_magnitudes:
        class_idx = magnitude_to_class(mag, class_mapping)
        print(f"  Magnitude {mag} -> Class {class_idx}")
    
    print(f"\nUtility functions test completed successfully!")