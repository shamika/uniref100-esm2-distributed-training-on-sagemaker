import torch
from torch.utils.data import Dataset
import boto3
from boto3.dynamodb.conditions import Key

class DynamoDBDataset(Dataset):
    def __init__(self, table_name="uniref100-esm2-tokenized", total_items=400000, region='us-east-1'):
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        self.table = self.dynamodb.Table(table_name)
        self.total_items = total_items  # Total number of items in the table

    def __len__(self):
        return self.total_items

    def __getitem__(self, idx):
        response = self.table.get_item(Key={'sequence_id': idx})
        item = response['Item']
        
        # Convert decimal.Decimal to a list of integers using map
        input_ids = list(map(int, item['input_ids']))
        attention_mask = list(map(int, item['attention_mask']))
        
        # Convert lists to torch tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }