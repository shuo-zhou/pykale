"""
Example demonstrating how to use PyKale's base dataset classes.

This example shows how to create custom datasets that inherit from PyKale's
base dataset classes, gaining common functionality like metadata management,
standardized interfaces, and consistent string representation.
"""

from kale.loaddata.base_dataset import BaseDataset, BaseTorchDataset, BaseGraphDataset


class SimpleDataset(BaseDataset):
    """A simple dataset example using BaseDataset."""
    
    def __init__(self, data, **kwargs):
        super().__init__(**kwargs)
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


class TorchCompatibleDataset(BaseTorchDataset):
    """A PyTorch-compatible dataset using BaseTorchDataset."""
    
    def __init__(self, size=100, **kwargs):
        super().__init__(**kwargs)
        # Generate some mock data
        import random
        self.data = [(random.random(), random.randint(0, 9)) for _ in range(size)]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


class GraphDataset(BaseGraphDataset):
    """A graph dataset using BaseGraphDataset."""
    
    def __init__(self, num_graphs=50, **kwargs):
        super().__init__(**kwargs)
        # Generate some mock graph data
        self.graphs = [f"graph_{i}" for i in range(num_graphs)]
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, index):
        return self.graphs[index]


def main():
    """Demonstrate usage of the base dataset classes."""
    
    print("PyKale Base Dataset Classes Example")
    print("=" * 40)
    
    # Example 1: Simple dataset
    print("\n1. Simple Dataset Example:")
    simple_data = [1, 2, 3, 4, 5]
    dataset1 = SimpleDataset(
        data=simple_data,
        name="SimpleExample",
        root="/tmp/simple",
        description="A basic dataset example"
    )
    
    print(f"Dataset: {dataset1}")
    print(f"Length: {len(dataset1)}")
    print(f"First item: {dataset1[0]}")
    print(f"Name: {dataset1.name}")
    print(f"Root: {dataset1.root}")
    print(f"Description: {dataset1.get_metadata('description')}")
    
    # Add some metadata
    dataset1.set_metadata("version", "1.0")
    dataset1.set_metadata("created_by", "PyKale Example")
    print(f"Version: {dataset1.get_metadata('version')}")
    print(f"All metadata: {dataset1.metadata}")
    
    # Example 2: PyTorch-compatible dataset
    print("\n2. PyTorch-Compatible Dataset Example:")
    dataset2 = TorchCompatibleDataset(
        size=10,
        name="TorchExample",
        root="/tmp/torch",
        task="classification"
    )
    
    print(f"Dataset: {dataset2}")
    print(f"Length: {len(dataset2)}")
    print(f"Sample data: {dataset2[0]}")
    print(f"Task: {dataset2.get_metadata('task')}")
    
    # This dataset can be used with PyTorch DataLoader
    try:
        import torch
        from torch.utils.data import DataLoader
        
        # Create a DataLoader (though we won't iterate due to random data)
        dataloader = DataLoader(dataset2, batch_size=2, shuffle=True)
        print(f"Can create PyTorch DataLoader: ✓")
        
        # Get one batch to demonstrate
        for batch in dataloader:
            print(f"Batch shape: {len(batch)} items")
            break
            
    except ImportError:
        print("PyTorch not available for DataLoader demo")
    
    # Example 3: Graph dataset
    print("\n3. Graph Dataset Example:")
    dataset3 = GraphDataset(
        num_graphs=5,
        num_classes=3,
        name="GraphExample",
        root="/tmp/graphs",
        graph_type="molecular"
    )
    
    print(f"Dataset: {dataset3}")
    print(f"Length: {len(dataset3)}")
    print(f"Number of classes: {dataset3.num_classes}")
    print(f"Graph type: {dataset3.get_metadata('graph_type')}")
    print(f"Sample graph: {dataset3[0]}")
    
    # Demonstrate metadata updates
    dataset3.update_metadata({
        "edge_types": ["single", "double", "aromatic"],
        "max_nodes": 50,
        "preprocessing": "normalized"
    })
    print(f"Updated metadata: {dataset3.get_metadata('edge_types')}")
    
    # Example 4: Dataset comparison
    print("\n4. Dataset Comparison:")
    datasets = [dataset1, dataset2, dataset3]
    
    for i, dataset in enumerate(datasets, 1):
        print(f"Dataset {i}:")
        print(f"  Type: {type(dataset).__name__}")
        print(f"  Length: {len(dataset)}")
        print(f"  Name: {dataset.name}")
        print(f"  Root: {dataset.root}")
        print(f"  Metadata keys: {list(dataset.metadata.keys())}")
        print(f"  String representation: {dataset}")
        print()
    
    print("Example completed successfully!")


if __name__ == "__main__":
    main()