# script to split TheDonald_Comments csv files into monthly split documents
import os, csv

n = 0
failed = 0
data_per_month = dict()

filenames = [
    '../../Data/RandomReddit/sample.csv'
]

for filename in filenames:
    with open(filename, 'r') as infile:
        reader = csv.DictReader(infile)
        data = [instance for instance in reader]

    for instance in data:
        if instance['created_utc'] != None and len(instance['created_utc']) == 10:
            month = instance['created_utc'][:-3]
            if month not in data_per_month:
                data_per_month[month] = []
            data_per_month[month].append(instance)
            n += 1
        else:
            failed += 1

for month, instances in data_per_month.items():

    filepath = f'../../Data/RandomReddit/{month}.csv' 
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a') as outfile:
        header = (['', 'text', 'id', 'author', 'created_utc', 'url']) 
        writer = csv.DictWriter(outfile, fieldnames=header)
        if not file_exists:
            writer.writeheader() 
        for instance in instances:
            writer.writerow(instance)
    print(month, '\t', len(instances))
    
print('Total number of comments =', n)
print('Total number of not included empty-date comments =', failed)