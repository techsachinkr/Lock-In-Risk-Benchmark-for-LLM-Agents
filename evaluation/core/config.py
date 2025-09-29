"""Configuration utilities"""

class Config:
    """Configuration class with merge support"""
    
    def __init__(self, data=None):
        self._data = data or {}
    
    def merge(self, other):
        if isinstance(other, dict):
            self._data.update(other)
        elif isinstance(other, Config):
            self._data.update(other._data)
        return self
    
    def __getattr__(self, name):
        return self._data.get(name)
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __setitem__(self, key, value):
        self._data[key] = value
