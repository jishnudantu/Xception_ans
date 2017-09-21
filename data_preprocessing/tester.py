import json
dataset = {}
with open('data_prepro.json','r') as data_file:
    data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]
print(dataset['unique_img_train'][:10])
