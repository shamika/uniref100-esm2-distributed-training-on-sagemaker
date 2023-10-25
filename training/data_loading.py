import argparse
import glob
from timeit import default_timer as timer
import os
import torch

import pandas as pd

def get_file_index_map(args):
    data_path = args.training_dir
    files = glob.glob(data_path + "/*")

    current_end_line_num = -1

    file_index_map_list = []
    for file in files:
        df = pd.read_parquet(file, engine='pyarrow')
        
        num_sequences = df.shape[0]
        start_line_num = current_end_line_num + 1
        end_line_num = start_line_num + num_sequences - 1
        
        current_end_line_num = end_line_num
        file_index_map_list.append({
            'file_name': os.path.basename(file),
            'num_sequences': num_sequences,
            'start_line_num': start_line_num,
            'end_line_num': end_line_num
        })
    file_index_map_df = pd.DataFrame(file_index_map_list)
    return file_index_map_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_dir", default="/Users/sariyawa/myworkspace/non-backup/source/uniref100-esm2-distributed-training-on-sagemaker/sample-data/torkenized-10-v1/trainx", help="Learning rate to use for training."
    )
    args, _ = parser.parse_known_args()
    file_index_map_df = get_file_index_map(args)
    print(file_index_map_df)
    index = 1
    filter = (file_index_map_df['start_line_num'] <= index) & (file_index_map_df['end_line_num'] >= index)

    index_file_row = file_index_map_df.loc[filter]

    item_file_df = pd.read_parquet(os.path.join(args.training_dir, index_file_row["file_name"].item()), engine='pyarrow')
    
    item_local_id = index - index_file_row["start_line_num"].item()
    
    item_row = item_file_df.iloc[item_local_id]

    input_ids = torch.from_numpy(item_row['input_ids'])
    attention_mask = torch.from_numpy(item_row['attention_mask'])

    sample = {'input_ids':input_ids, 'attention_mask':attention_mask}

    print(sample)