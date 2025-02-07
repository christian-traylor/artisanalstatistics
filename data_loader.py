import csv

def csv_loader(filename, pandas=False):
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader, None)
        if header is None:
            raise ValueError("File is empty or missing a header row.")
        data = list(csv_reader)
    return header, data