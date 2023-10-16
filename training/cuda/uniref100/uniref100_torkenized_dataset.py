import os
import torch
import pandas as pd


class Uniref100TorkenizedDataset(torch.utils.data.Dataset):
    #'Characterizes a dataset for PyTorch'
    # file_index_map_path - Should be a csv with schema : file_name,num_sequences,start_line,end_line

    def __init__(self, data_dir, file_index_map_path):    
        #'Initialization'
        self.data_dir = data_dir
        self.file_index_map_df = pd.read_csv(file_index_map_path)

    def __len__(self):
        # 'Denotes the total number of samples'
        return self.file_index_map_df[-1:]["end_line"].item()
        
    def __getitem__(self, index):
        _filter = (self.file_index_map_df['start_line'] <= index) & (self.file_index_map_df['end_line'] >= index)
        index_file_row = self.file_index_map_df.loc[_filter]
        
        item_file_df = pd.read_parquet(os.path.join(self.data_dir, index_file_row["file_name"].item()), engine='pyarrow')
        
        item_local_id = index - index_file_row["start_line"].item()
        
        item_row = item_file_df.iloc[item_local_id]
        
        input_ids = torch.from_numpy(item_row['input_ids'].copy())
        attention_mask = torch.from_numpy(item_row['attention_mask'].copy())

        sample = {'input_ids':input_ids, 'attention_mask':attention_mask}
        
        return sample