import os
from random import random

a = 0.90 # separator factor (1 to put everything into the train dataset)

def linecount(file_path):
	with open(file_path, 'r') as f:
		return sum(1 for _ in f)

if __name__ == "__main__":
	datasets_dir = "datasets"
	os.makedirs("datasets", exist_ok=True)

	source_file		= "data.csv"
	train_file		= os.path.join(datasets_dir, "train.csv")
	predict_file	= os.path.join(datasets_dir, "predict.csv")

	with open(source_file, 'r') as data, open(train_file, 'w') as train, open(predict_file, 'w') as predict:
		for line in data:
			if random() < a:
				train.write(line)
			else:
				predict.write(line)

	print(f"length train file:\t{linecount(train_file)}")
	print(f"length predict file:\t{linecount(predict_file)}")
	print(f"factor of separation :\t{a}")
