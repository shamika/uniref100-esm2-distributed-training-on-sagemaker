import os
import torch
import pandas as pd


import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)  # Mark as recently used
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Remove least recently used item

class Uniref100TorkenizedDataset(Dataset):
    def __init__(self, data_dir, file_index_map_path, cache_size=10):
        self.data_dir = data_dir
        self.file_index_map_df = pd.read_csv(file_index_map_path)
        self.file_cache = LRUCache(cache_size)

    def __len__(self):
        return self.file_index_map_df.iloc[-1]["end_line"]

    def __getitem__(self, index):
        file_name, item_local_id = self.get_file_info(index)
        item_file_df = self.file_cache.get(file_name)
        if item_file_df is None:
            #print(f"[CACHE] MISS")
            item_file_df = pd.read_parquet(os.path.join(self.data_dir, file_name), engine='pyarrow')
            self.file_cache.put(file_name, item_file_df)
        #else:
            #print("[CACHE] HIT")
        item_row = item_file_df.iloc[item_local_id]
        return {
            'input_ids': torch.from_numpy(item_row['input_ids'].copy()),
            'attention_mask': torch.from_numpy(item_row['attention_mask'].copy())
        }

    def get_file_info(self, index):
        _filter = (self.file_index_map_df['start_line'] <= index) & (self.file_index_map_df['end_line'] >= index)
        index_file_row = self.file_index_map_df.loc[_filter].iloc[0]
        file_name = index_file_row["file_name"]
        item_local_id = index - index_file_row["start_line"]
        return file_name, item_local_id