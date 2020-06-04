import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import pandas as pd
from job_posting import JobPosting, JobPostingsDataset
import random
import network
from pycm import ConfusionMatrix


CSV_FILENAME = "dataset.csv"
CSV_FILE_DELIMITER = ","
EPOCHS_COUNT = 40
BATCH_SIZE = 32
TRAINING_DATASET_SIZE_RATIO = 0.85
LEARNING_RATE = 0.00005


def get_batch_data_tensor(index, batch_size=BATCH_SIZE):
	return torch.tensor([training_dataset[i].get_data_list() for i in range(index, index + batch_size)])


def get_batch_target_tensor(index, batch_size=BATCH_SIZE):
	return torch.tensor([training_dataset[i].get_target_list() for i in range(index, index + batch_size)])


def prepare_training_and_validation_datasets(training_set_size_ratio=TRAINING_DATASET_SIZE_RATIO, batch_size=BATCH_SIZE):
	global training_dataset
	global validation_dataset
	global training_dataset_size
	global validation_dataset_size
	global batches_count
	global job_postings
	random.shuffle(job_postings)
	training_dataset_size = (int(len(job_postings) * training_set_size_ratio) // batch_size) * batch_size
	job_postings = JobPostingsDataset(job_postings)
	job_postings.prepare_all_text_vectorizers()
	training_dataset = [job_postings[i] for i in range(0, training_dataset_size)]
	validation_dataset = [job_postings[i] for i in range(training_dataset_size, len(job_postings))]
	validation_dataset_size = len(validation_dataset)
	batches_count = training_dataset_size // batch_size


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
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
	# for m in model._modules:
		# network.normal_init(model._modules[m], 0, 1)


def validate():
	global input_labels
	global predicted_labels
	input_labels = []
	predicted_labels = []
	model.eval()
	total_true_positives = 0
	with torch.no_grad():
		for j in range(validation_dataset_size):
			network_output = model(validation_dataset[j].get_data_tensor().float().to(device))
			network_output_index = int(network_output.argmax(dim=0, keepdim=False))
			if network_output_index == validation_dataset[j].get_target():
				total_true_positives += 1
			input_labels.append(validation_dataset[j].get_target())
			predicted_labels.append(network_output_index)
	accuracy = total_true_positives / validation_dataset_size
	print("Accuracy:", accuracy)


def print_confusion_matrix(input_labels, predicted_labels):
	cm = ConfusionMatrix(input_labels, predicted_labels)
	print(cm)


def train_and_validate_all(epochs_count=EPOCHS_COUNT, batch_size=BATCH_SIZE):
	model.train()
	for i in range(epochs_count):
		training_loss = 0
		for j in range(batches_count):
			network_output = model(get_batch_data_tensor(j).float().to(device))
			optimizer.zero_grad()
			loss_value = network.loss(network_output, get_batch_target_tensor(j))
			training_loss += loss_value.item()
			loss_value.backward()
			optimizer.step()
		training_loss /= batches_count
		print("Epoch number:", i + 1)
		print("Training loss:", training_loss)
		validate()
		print("---------------------")


if __name__ == "__main__":
	initialize_network()
	read_job_postings_data_from_csv()
	prepare_training_and_validation_datasets()
	train_and_validate_all()
	print_confusion_matrix(input_labels, predicted_labels)
