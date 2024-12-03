import os
from random import random
import argparse
import shelve
import sys

parser = argparse.ArgumentParser(description="A program that splits a DataSet")
parser.add_argument("-a", "--split_factor", type=float, help="split factor (default value 0.9)")

def linecount(file_path):
	with open(file_path, 'r') as f:
		return sum(1 for _ in f)

def main():
	a = 0.50 # default split
	datasets_dir = "datasets"
	args = parser.parse_args()
	with shelve.open(".save_parameters") as file:
		if args.split_factor is not None:
			a = args.split_factor
	os.makedirs(datasets_dir, exist_ok=True)

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

if __name__ == "__main__":
	main()