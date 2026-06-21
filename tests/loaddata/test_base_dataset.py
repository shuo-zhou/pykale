"""
Tests for the base dataset classes.
"""

import pytest

from kale.loaddata.base_dataset import BaseDataset, BaseTorchDataset, BaseGraphDataset


class MockDataset(BaseDataset):
    """Mock implementation of BaseDataset for testing."""
    
    def __init__(self, data=None, **kwargs):
        super().__init__(**kwargs)
        self.data = data or list(range(10))  # Default to 10 items
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


class MockTorchDataset(BaseTorchDataset):
    """Mock implementation of BaseTorchDataset for testing."""
    
    def __init__(self, data=None, **kwargs):
        super().__init__(**kwargs)
        self.data = data or list(range(5))  # Default to 5 items
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


class MockGraphDataset(BaseGraphDataset):
    """Mock implementation of BaseGraphDataset for testing."""
    
    def __init__(self, data=None, **kwargs):
        super().__init__(**kwargs)
        self.data = data or list(range(8))  # Default to 8 items
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


class TestBaseDataset:
    """Test cases for BaseDataset class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        dataset = MockDataset()
        assert dataset.name is None
        assert dataset.root is None
        assert dataset.metadata == {}
    
    def test_initialization_with_params(self):
        """Test initialization with parameters."""
        dataset = MockDataset(name="test_dataset", root="/path/to/data")
        assert dataset.name == "test_dataset"
        assert dataset.root == "/path/to/data"
        assert dataset.metadata == {}
    
    def test_initialization_with_kwargs(self):
        """Test initialization with additional kwargs."""
        dataset = MockDataset(
            name="test",
            root="/path",
            num_samples=100,
            version="1.0"
        )
        assert dataset.name == "test"
        assert dataset.root == "/path"
        assert dataset.get_metadata("num_samples") == 100
        assert dataset.get_metadata("version") == "1.0"
    
    def test_len_implementation(self):
        """Test __len__ method implementation."""
        data = [1, 2, 3, 4, 5]
        dataset = MockDataset(data=data)
        assert len(dataset) == 5
    
    def test_getitem_implementation(self):
        """Test __getitem__ method implementation."""
        data = ["a", "b", "c"]
        dataset = MockDataset(data=data)
        assert dataset[0] == "a"
        assert dataset[1] == "b"
        assert dataset[2] == "c"
    
    def test_metadata_operations(self):
        """Test metadata getter and setter operations."""
        dataset = MockDataset()
        
        # Test getting non-existent key
        assert dataset.get_metadata("nonexistent") is None
        assert dataset.get_metadata("nonexistent", "default") == "default"
        
        # Test setting metadata
        dataset.set_metadata("key1", "value1")
        assert dataset.get_metadata("key1") == "value1"
        
        # Test updating metadata
        dataset.update_metadata({"key2": "value2", "key3": 123})
        assert dataset.get_metadata("key2") == "value2"
        assert dataset.get_metadata("key3") == 123
        
        # Test metadata property
        metadata = dataset.metadata
        assert metadata["key1"] == "value1"
        assert metadata["key2"] == "value2"
        assert metadata["key3"] == 123
        
        # Ensure it's a copy
        metadata["key4"] = "value4"
        assert dataset.get_metadata("key4") is None
    
    def test_repr(self):
        """Test string representation."""
        dataset = MockDataset(name="test_data", root="/data")
        repr_str = repr(dataset)
        assert "MockDataset" in repr_str
        assert "length=10" in repr_str
        assert "name=test_data" in repr_str
        assert "root=/data" in repr_str
    
    def test_repr_no_name_no_root(self):
        """Test string representation without name and root."""
        dataset = MockDataset()
        repr_str = repr(dataset)
        assert "MockDataset" in repr_str
        assert "length=10" in repr_str
        assert "name=" not in repr_str or "name=None" not in repr_str
        assert "root=" not in repr_str or "root=None" not in repr_str
    
    def test_abstract_methods(self):
        """Test that BaseDataset cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseDataset()


class TestBaseTorchDataset:
    """Test cases for BaseTorchDataset class."""
    
    def test_initialization(self):
        """Test BaseTorchDataset initialization."""
        dataset = MockTorchDataset(name="torch_test", root="/torch/data")
        assert dataset.name == "torch_test"
        assert dataset.root == "/torch/data"
        assert len(dataset) == 5
    
    def test_dataset_interface(self):
        """Test PyTorch Dataset interface compliance."""
        dataset = MockTorchDataset()
        assert hasattr(dataset, "__len__")
        assert hasattr(dataset, "__getitem__")
        assert len(dataset) == 5
        assert dataset[0] == 0
        assert dataset[4] == 4
    
    def test_inheritance(self):
        """Test inheritance from BaseDataset."""
        dataset = MockTorchDataset()
        assert isinstance(dataset, BaseDataset)
        assert isinstance(dataset, BaseTorchDataset)
        
        # Test inherited functionality
        dataset.set_metadata("test_key", "test_value")
        assert dataset.get_metadata("test_key") == "test_value"


class TestBaseGraphDataset:
    """Test cases for BaseGraphDataset class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        dataset = MockGraphDataset()
        assert dataset.num_classes is None
        assert len(dataset) == 8
    
    def test_initialization_with_num_classes(self):
        """Test initialization with num_classes."""
        dataset = MockGraphDataset(num_classes=5, name="graph_test")
        assert dataset.num_classes == 5
        assert dataset.name == "graph_test"
    
    def test_num_classes_property(self):
        """Test num_classes property getter and setter."""
        dataset = MockGraphDataset()
        assert dataset.num_classes is None
        
        dataset.num_classes = 10
        assert dataset.num_classes == 10
        
        dataset.num_classes = 3
        assert dataset.num_classes == 3
    
    def test_inheritance(self):
        """Test inheritance from BaseDataset."""
        dataset = MockGraphDataset(num_classes=7)
        assert isinstance(dataset, BaseDataset)
        assert isinstance(dataset, BaseGraphDataset)
        
        # Test inherited functionality
        dataset.set_metadata("graph_type", "molecular")
        assert dataset.get_metadata("graph_type") == "molecular"
    
    def test_kwargs_handling(self):
        """Test kwargs handling in BaseGraphDataset."""
        dataset = MockGraphDataset(
            num_classes=4,
            name="test_graph",
            root="/graph/data",
            edge_type="directed"
        )
        assert dataset.num_classes == 4
        assert dataset.name == "test_graph"
        assert dataset.root == "/graph/data"
        assert dataset.get_metadata("edge_type") == "directed"


class TestDatasetIntegration:
    """Integration tests for dataset base classes."""
    
    def test_multiple_inheritance_patterns(self):
        """Test that datasets can work with different inheritance patterns."""
        torch_dataset = MockTorchDataset(name="torch_test")
        graph_dataset = MockGraphDataset(num_classes=5, name="graph_test")
        
        # Both should support basic dataset operations
        assert len(torch_dataset) == 5
        assert len(graph_dataset) == 8
        
        assert torch_dataset[2] == 2
        assert graph_dataset[3] == 3
        
        # Both should support metadata operations
        torch_dataset.set_metadata("type", "torch")
        graph_dataset.set_metadata("type", "graph")
        
        assert torch_dataset.get_metadata("type") == "torch"
        assert graph_dataset.get_metadata("type") == "graph"
    
    def test_repr_consistency(self):
        """Test that repr works consistently across different base classes."""
        torch_dataset = MockTorchDataset(name="torch_test")
        graph_dataset = MockGraphDataset(name="graph_test")
        
        torch_repr = repr(torch_dataset)
        graph_repr = repr(graph_dataset)
        
        assert "MockTorchDataset" in torch_repr
        assert "MockGraphDataset" in graph_repr
        assert "name=torch_test" in torch_repr
        assert "name=graph_test" in graph_repr