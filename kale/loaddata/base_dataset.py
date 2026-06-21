"""
Base dataset class with common functionality for PyKale datasets.

This module provides a common base class that captures shared attributes and methods
across various dataset implementations in PyKale, promoting code reuse while maintaining
backward compatibility.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union


class BaseDataset(ABC):
    """
    Base dataset class that provides common functionality for PyKale datasets.
    
    This class establishes a common interface and shared functionality that can be
    inherited by specific dataset implementations. It focuses on the most universally
    shared attributes and methods to maximize reusability while maintaining flexibility.
    
    Args:
        name (str, optional): Name identifier for the dataset. Defaults to None.
        root (str, optional): Root directory for dataset storage. Defaults to None.
        
    Attributes:
        name (str): Dataset name identifier
        root (str): Root directory path
        _metadata (dict): Dictionary to store dataset metadata
    """
    
    def __init__(self, name: Optional[str] = None, root: Optional[str] = None, **kwargs):
        self.name = name
        self.root = root
        self._metadata = {}
        
        # Store any additional kwargs in metadata for flexibility
        for key, value in kwargs.items():
            if not key.startswith('_'):  # Avoid overriding private attributes
                self._metadata[key] = value
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        pass
    
    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """
        Retrieve a sample from the dataset at the given index.
        
        Args:
            index (int): Index of the sample to retrieve
            
        Returns:
            Any: The data sample at the specified index
        """
        pass
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Retrieve metadata value by key.
        
        Args:
            key (str): Metadata key
            default (Any, optional): Default value if key not found. Defaults to None.
            
        Returns:
            Any: Metadata value or default
        """
        return self._metadata.get(key, default)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set a metadata key-value pair.
        
        Args:
            key (str): Metadata key
            value (Any): Metadata value
        """
        self._metadata[key] = value
    
    def update_metadata(self, metadata_dict: dict) -> None:
        """
        Update metadata with a dictionary of key-value pairs.
        
        Args:
            metadata_dict (dict): Dictionary of metadata to update
        """
        self._metadata.update(metadata_dict)
    
    @property
    def metadata(self) -> dict:
        """
        Get a copy of all metadata.
        
        Returns:
            dict: Copy of metadata dictionary
        """
        return self._metadata.copy()
    
    def __repr__(self) -> str:
        """
        Return string representation of the dataset.
        
        Returns:
            str: String representation
        """
        class_name = self.__class__.__name__
        name_str = f", name={self.name}" if self.name else ""
        root_str = f", root={self.root}" if self.root else ""
        return f"{class_name}(length={len(self)}{name_str}{root_str})"


class BaseTorchDataset(BaseDataset):
    """
    Base dataset class that extends BaseDataset with PyTorch Dataset functionality.
    
    This class provides a bridge between PyKale's BaseDataset and PyTorch's Dataset,
    allowing datasets to benefit from both PyTorch's data loading infrastructure and
    PyKale's common dataset functionality.
    """
    
    def __init__(self, name: Optional[str] = None, root: Optional[str] = None, **kwargs):
        super().__init__(name=name, root=root, **kwargs)
    
    # The PyTorch Dataset interface is satisfied by the abstract methods
    # __len__ and __getitem__ from BaseDataset


class BaseGraphDataset(BaseDataset):
    """
    Base dataset class for graph-based datasets.
    
    This class extends BaseDataset with common functionality specific to graph datasets,
    such as those used with PyTorch Geometric.
    
    Args:
        num_classes (int, optional): Number of classes in the dataset. Defaults to None.
        **kwargs: Additional arguments passed to BaseDataset
    """
    
    def __init__(self, num_classes: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self._num_classes = num_classes
    
    @property
    def num_classes(self) -> Optional[int]:
        """
        Get the number of classes in the dataset.
        
        Returns:
            Optional[int]: Number of classes, or None if not specified
        """
        return self._num_classes
    
    @num_classes.setter
    def num_classes(self, value: int) -> None:
        """
        Set the number of classes in the dataset.
        
        Args:
            value (int): Number of classes
        """
        self._num_classes = value