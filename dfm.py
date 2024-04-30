import pandas as pd

class DFM:

    def __init__(self, data: pd.DataFrame, r: int, extractor: str='PC') -> None:
        
        self.data = data
        self.T, self.N = data.shape
        self.r = r
        self.extractor = extractor