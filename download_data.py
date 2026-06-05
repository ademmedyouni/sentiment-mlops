from datasets import load_dataset
import os
os.makedirs('data/raw', exist_ok=True)
ds = load_dataset('imdb')
ds['train'].to_csv('data/raw/train.csv', index=False)
ds['test'].to_csv('data/raw/test.csv', index=False)
print('train:', len(ds['train']), 'rows')
print('test: ', len(ds['test']), 'rows')
print('Done!')