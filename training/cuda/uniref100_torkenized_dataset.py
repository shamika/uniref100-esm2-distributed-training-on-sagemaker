import os
import torch
import pandas as pd


class Uniref100TorkenizedDataset(torch.utils.data.Dataset):
    #'Characterizes a dataset for PyTorch'
    # file_index_map_path - Should be a csv with schema : file_name,num_sequences,start_line,end_line

    def __init__(self, data_dir, file_index_map_path):    
        #'Initialization'
        self.data_dir = data_dir
        print("Exact index file path is [{}]".formatfile_index_map_path))
        self.file_index_map = pd.read_csv(file_index_map_path)

    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.file_index_map.shape[0])
        
    def __getitem__(self, index):
        filter = (self.file_index_map_df['start_line'] <= index) & (self.file_index_map_df['end_line'] >= index)
        index_file_row = self.file_index_map_df.loc[filter]
        index_file_df = pd.read_parquet(os.path.join(self.data_dir, index_file_row["file_name"]), engine='pyarrow')
        
        local_id = index - index_file_row["start_line"]
        
        input_ids = torch.from_numpy(index_file_df['input_ids'][local_id])
        attention_mask = torch.from_numpy(index_file_df['attention_mask'][local_id])

        sample = {'input_ids':input_ids, 'attention_mask':attention_mask}
        return sample