"""
Integration tests showing how existing datasets benefit from the new base classes.
"""

import pytest
import torch

from kale.loaddata.base_dataset import BaseDataset, BaseTorchDataset, BaseGraphDataset


class TestDatasetIntegration:
    """Test integration of base classes with dataset patterns similar to existing PyKale datasets."""
    
    def test_tdc_style_dataset_with_base_class(self):
        """Test a TDC-style dataset using BaseTorchDataset."""
        
        class MockTDCDataset(BaseTorchDataset):
            """Mock implementation of TDC-style dataset with base class benefits."""
            
            def __init__(self, name: str, split="train", path="./data", mode="cnn_cnn", **kwargs):
                super().__init__(name=name, root=path, split=split, mode=mode, **kwargs)
                
                # Mock data similar to TDC datasets
                self.data = {"Drug": ["CCO", "CCC"], "Target": ["PROTEIN1", "PROTEIN2"], "Y": [1.0, 2.0]}
                self.mode = mode.lower()
            
            def __len__(self):
                return len(self.data["Y"])
            
            def __getitem__(self, idx):
                drug = self.data["Drug"][idx]
                protein = self.data["Target"][idx]
                label = self.data["Y"][idx]
                return drug, protein, label
        
        # Test the dataset
        dataset = MockTDCDataset(name="BindingDB_Test", split="train", path="./test_data")
        
        # Test inherited functionality
        assert len(dataset) == 2
        assert dataset.name == "BindingDB_Test"
        assert dataset.root == "./test_data"
        assert dataset.get_metadata("split") == "train"
        assert dataset.get_metadata("mode") == "cnn_cnn"
        
        # Test dataset functionality
        drug, protein, label = dataset[0]
        assert drug == "CCO"
        assert protein == "PROTEIN1"
        assert label == 1.0
        
        # Test metadata operations
        dataset.set_metadata("version", "1.0")
        assert dataset.get_metadata("version") == "1.0"
        
        # Test repr
        repr_str = repr(dataset)
        assert "MockTDCDataset" in repr_str
        assert "length=2" in repr_str
        assert "name=BindingDB_Test" in repr_str
    
    def test_multiomics_style_dataset_with_base_class(self):
        """Test a multiomics-style dataset using BaseGraphDataset."""
        
        class MockMultiomicsDataset(BaseGraphDataset):
            """Mock implementation of multiomics-style dataset with base class benefits."""
            
            def __init__(self, num_modalities: int, num_classes: int, root: str = "./data", **kwargs):
                super().__init__(
                    num_classes=num_classes,
                    name=f"Multiomics_{num_modalities}mod",
                    root=root,
                    num_modalities=num_modalities,
                    **kwargs
                )
                
                self._num_modalities = num_modalities
                # Mock data list for modalities
                self._data_list = [f"modality_{i}_data" for i in range(num_modalities)]
            
            def __len__(self):
                return 1  # Similar to original implementation
            
            def __getitem__(self, index):
                return self._data_list
            
            @property
            def num_modalities(self):
                return self._num_modalities
        
        # Test the dataset
        dataset = MockMultiomicsDataset(num_modalities=3, num_classes=5, root="./multiomics_data")
        
        # Test inherited functionality from BaseGraphDataset
        assert dataset.num_classes == 5
        assert dataset.name == "Multiomics_3mod"
        assert dataset.root == "./multiomics_data"
        assert dataset.get_metadata("num_modalities") == 3
        
        # Test dataset-specific functionality
        assert dataset.num_modalities == 3
        assert len(dataset) == 1
        data_list = dataset[0]
        assert len(data_list) == 3
        assert data_list[0] == "modality_0_data"
        
        # Test metadata operations
        dataset.set_metadata("random_split", True)
        dataset.set_metadata("train_size", 0.8)
        assert dataset.get_metadata("random_split") is True
        assert dataset.get_metadata("train_size") == 0.8
        
        # Test repr
        repr_str = repr(dataset)
        assert "MockMultiomicsDataset" in repr_str
        assert "length=1" in repr_str
        assert "name=Multiomics_3mod" in repr_str
    
    def test_polypharmacy_style_dataset_with_base_class(self):
        """Test a polypharmacy-style dataset using BaseTorchDataset."""
        
        class MockPolypharmacyDataset(BaseTorchDataset):
            """Mock implementation of polypharmacy-style dataset with base class benefits."""
            
            def __init__(self, url: str, name: str, mode: str = "train", **kwargs):
                super().__init__(name=name, url=url, mode=mode, **kwargs)
                
                # Mock data similar to polypharmacy datasets
                self.edge_index = torch.tensor([[0, 1], [1, 2]])
                self.edge_type = torch.tensor([1, 2])
                self.edge_type_range = torch.tensor([0, 1, 2])
                
                if mode == "train":
                    self.protein_feat = torch.randn(10, 5)
                    self.drug_feat = torch.randn(8, 3)
            
            def __len__(self):
                return 1  # Similar to original implementation
            
            def __getitem__(self, idx):
                return self.edge_index, self.edge_type, self.edge_type_range
        
        # Test the dataset
        dataset = MockPolypharmacyDataset(
            url="https://example.com/data.pt",
            name="TestPolypharmacy",
            mode="train",
            root="./poly_data"
        )
        
        # Test inherited functionality
        assert dataset.name == "TestPolypharmacy"
        assert dataset.root == "./poly_data"
        assert dataset.get_metadata("url") == "https://example.com/data.pt"
        assert dataset.get_metadata("mode") == "train"
        
        # Test dataset functionality
        assert len(dataset) == 1
        edge_index, edge_type, edge_type_range = dataset[0]
        assert edge_index.shape == (2, 2)
        assert edge_type.shape == (2,)
        assert edge_type_range.shape == (3,)
        
        # Test training mode specific attributes
        assert hasattr(dataset, 'protein_feat')
        assert hasattr(dataset, 'drug_feat')
        assert dataset.protein_feat.shape == (10, 5)
        assert dataset.drug_feat.shape == (8, 3)
    
    def test_avmnist_style_dataset_with_base_class(self):
        """Test an AVMNIST-style dataset using BaseDataset."""
        
        class MockAVMNISTDataset(BaseDataset):
            """Mock implementation of AVMNIST-style dataset with base class benefits."""
            
            def __init__(self, data_dir: str, batch_size: int = 40, **kwargs):
                super().__init__(name="AVMNIST", root=data_dir, batch_size=batch_size, **kwargs)
                
                # Mock multimodal data
                self.train_data = [(torch.randn(1, 28, 28), torch.randn(1, 112, 112), i % 10) for i in range(100)]
                self.test_data = [(torch.randn(1, 28, 28), torch.randn(1, 112, 112), i % 10) for i in range(20)]
            
            def __len__(self):
                return len(self.train_data)
            
            def __getitem__(self, idx):
                return self.train_data[idx]
            
            def get_test_data(self):
                return self.test_data
        
        # Test the dataset
        dataset = MockAVMNISTDataset(
            data_dir="./avmnist_data",
            batch_size=32,
            flatten_audio=False,
            normalize_image=True
        )
        
        # Test inherited functionality
        assert dataset.name == "AVMNIST"
        assert dataset.root == "./avmnist_data"
        assert dataset.get_metadata("batch_size") == 32
        assert dataset.get_metadata("flatten_audio") is False
        assert dataset.get_metadata("normalize_image") is True
        
        # Test dataset functionality
        assert len(dataset) == 100
        image, audio, label = dataset[5]
        assert image.shape == (1, 28, 28)
        assert audio.shape == (1, 112, 112)
        assert isinstance(label, int)
        
        # Test additional dataset methods
        test_data = dataset.get_test_data()
        assert len(test_data) == 20
    
    def test_cross_dataset_compatibility(self):
        """Test that different dataset types can work together."""
        
        # Create different types of datasets
        torch_dataset = TestDatasetIntegration.MockTDCDataset(self, name="test1", split="train")
        graph_dataset = TestDatasetIntegration.MockMultiomicsDataset(
            self, num_modalities=2, num_classes=3
        )
        
        # Verify they all support common interface
        datasets = [torch_dataset, graph_dataset]
        
        for dataset in datasets:
            # All should support length
            assert hasattr(dataset, "__len__")
            assert len(dataset) >= 0
            
            # All should support indexing
            assert hasattr(dataset, "__getitem__")
            
            # All should support metadata operations
            dataset.set_metadata("test_key", "test_value")
            assert dataset.get_metadata("test_key") == "test_value"
            
            # All should have string representation
            repr_str = repr(dataset)
            assert isinstance(repr_str, str)
            assert len(repr_str) > 0
    
    # Helper methods to avoid self reference issues
    def MockTDCDataset(self, **kwargs):
        class MockTDCDataset(BaseTorchDataset):
            def __init__(self, name: str, split="train", path="./data", mode="cnn_cnn", **kwargs):
                super().__init__(name=name, root=path, split=split, mode=mode, **kwargs)
                self.data = {"Drug": ["CCO"], "Target": ["PROTEIN1"], "Y": [1.0]}
                self.mode = mode.lower()
            
            def __len__(self):
                return len(self.data["Y"])
            
            def __getitem__(self, idx):
                return self.data["Drug"][idx], self.data["Target"][idx], self.data["Y"][idx]
        
        return MockTDCDataset(**kwargs)
    
    def MockMultiomicsDataset(self, **kwargs):
        class MockMultiomicsDataset(BaseGraphDataset):
            def __init__(self, num_modalities: int, num_classes: int, root: str = "./data", **kwargs):
                super().__init__(
                    num_classes=num_classes,
                    name=f"Multiomics_{num_modalities}mod",
                    root=root,
                    num_modalities=num_modalities,
                    **kwargs
                )
                self._num_modalities = num_modalities
                self._data_list = [f"modality_{i}_data" for i in range(num_modalities)]
            
            def __len__(self):
                return 1
            
            def __getitem__(self, index):
                return self._data_list
            
            @property
            def num_modalities(self):
                return self._num_modalities
        
        return MockMultiomicsDataset(**kwargs)