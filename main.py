import torch
from torch import nn, optim
import pandas as pd
from job_posting import JobPosting, JobPostingsDataset
import random
import network
from pycm import ConfusionMatrix
from skorch import NeuralNetClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score


CSV_FILENAME = "dataset.csv"
CSV_FILE_DELIMITER = ","
EPOCHS_COUNT = 20
LEARNING_RATE = 0.00001
KFOLD_PARTITIONS_COUNT = 5


def prepare_dataset():
	global job_postings
	global all_job_posting_data
	global all_job_posting_targets
	# random.shuffle(job_postings)
	job_postings = JobPostingsDataset(job_postings)
	job_postings.prepare_all_text_vectorizers()
	all_job_posting_data = torch.tensor([job_posting.get_data_list() for job_posting in job_postings]).float().to(device)
	all_job_posting_targets = torch.tensor([job_posting.get_target() for job_posting in job_postings]).float().to(device)


def read_job_postings_data_from_csv(csv_filename=CSV_FILENAME, csv_file_delimiter=CSV_FILE_DELIMITER):
	global job_postings
	job_postings = []
	csv_contents = pd.read_csv(csv_filename, delimiter=csv_file_delimiter, dtype=str)
	rows = len(csv_contents)
	cols = len(csv_contents.iloc[0])
	for i in range(rows):
		data = csv_contents.iloc[i]
		job_posting = JobPosting(data["title"], data["location"], data["department"], data["salary_range"], data["company_profile"], data["description"], data["requirements"],
								 data["benefits"], data["telecommuting"], data["has_company_logo"], data["has_questions"], data["employment_type"], data["required_experience"],
								 data["required_education"], data["industry"], data["function"], data["fraudulent"])
		job_postings.append(job_posting)


def initialize_network():
	global device
	global model
	global optimizer
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = network.Network().to(device)
	# for m in model._modules:
		# network.normal_init(model._modules[m], 0, 1)


if __name__ == "__main__":
	print("Initilizing network")
	initialize_network()
	print("Reading CSV file")
	read_job_postings_data_from_csv()
	print("Preparing dataset")
	prepare_dataset()
	print("Creating classifier")
	classifier = NeuralNetClassifier(network.Network, max_epochs=EPOCHS_COUNT, lr=LEARNING_RATE, train_split=None, criterion=torch.nn.MSELoss, optimizer=torch.optim.Adam)
	print("Training")
	recalls = cross_val_score(classifier, all_job_posting_data, all_job_posting_targets, cv=KFOLD_PARTITIONS_COUNT, verbose=1, scoring="recall")
	print("K-Fold cross validation recalls:", recalls)
