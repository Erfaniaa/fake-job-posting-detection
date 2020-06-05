import torch
import re
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
import numpy as np
import pandas as pd
from copy import deepcopy

SMALL_TEXT_VECTOR_SIZE = 10
MEDIUM_TEXT_VECTOR_SIZE = 25
LARGE_TEXT_VECTOR_SIZE = 50


class JobPosting:
	def __init__(self, title="", location="", department="", salary_range="", company_profile="",
				 description="", requirement="", benefit="", telecommuting="", has_company_logo="",
				 has_question="", employment_type="", required_experience="",
				 required_education="", industry="", function="", fraudulent=""):
		self.title = title if isinstance(title, str) else ""
		self.location = location if isinstance(location, str) else ""
		self.department = department if isinstance(department, str) else ""
		self.salary_range = [salary_range if isinstance(salary_range, str) else ""]
		self.company_profile = company_profile if isinstance(company_profile, str) else ""
		self.description = description if isinstance(description, str) else ""
		self.requirement = requirement if isinstance(requirement, str) else ""
		self.benefit = benefit if isinstance(benefit, str) else ""
		self.telecommuting = telecommuting if isinstance(telecommuting, str) else ""
		self.has_company_logo = has_company_logo if str(isinstance(has_company_logo, int)) else ""
		self.has_question = has_question if isinstance(has_question, str) else ""
		self.employment_type = employment_type if isinstance(employment_type, str) else ""
		self.required_experience = required_experience if isinstance(required_experience, str) else ""
		self.required_education = required_education if isinstance(required_education, str) else ""
		self.industry = industry if isinstance(industry, str) else ""
		self.function = function if isinstance(function, str) else ""
		self.fraudulent = fraudulent if isinstance(fraudulent, str) else ""

	def get_data_list(self):
		return [*self.title, *self.location, *self.department, *self.salary_range, *self.company_profile,
				*self.description, *self.requirement, *self.benefit, self.telecommuting, self.has_company_logo,
				self.has_company_logo, self.has_question, *self.employment_type, *self.required_experience,
				*self.required_education, *self.industry, *self.function]

	def get_target_list(self):
		if int(self.fraudulent) == 0:
			return [1.0, 0.0]
		else:
			return [0.0, 1.0]

	def get_target(self):
		return int(self.fraudulent)

	def get_data_tensor(self):
		return torch.tensor(self.get_data_list())


class JobPostingsDataset(torch.utils.data.Dataset):
	def __init__(self, job_postings_list=[]):
		self.job_postings_list = job_postings_list
		self.vectorized_job_postings_dict = {}

	def __len__(self):
		return len(self.job_postings_list)

	def prepare_all_text_vectorizers(self):
		self.title_vectorizer = TfidfVectorizer(max_features=MEDIUM_TEXT_VECTOR_SIZE)
		all_titles_list = [self.preprocess_text(job_posting.title) for job_posting in self.job_postings_list]
		self.title_vectorizer.fit_transform(all_titles_list)
		
		self.location_vectorizer = TfidfVectorizer(max_features=MEDIUM_TEXT_VECTOR_SIZE)
		all_locations_list = [self.preprocess_text(job_posting.location) for job_posting in self.job_postings_list]
		self.location_vectorizer.fit_transform(all_locations_list)
		
		self.department_vectorizer = TfidfVectorizer(max_features=SMALL_TEXT_VECTOR_SIZE)
		all_departments_list = [self.preprocess_text(job_posting.department) for job_posting in self.job_postings_list]
		self.department_vectorizer.fit_transform(all_departments_list)

		self.company_profile_vectorizer = TfidfVectorizer(max_features=LARGE_TEXT_VECTOR_SIZE)
		all_company_profiles_list = [self.preprocess_text(job_posting.company_profile) for job_posting in self.job_postings_list]
		self.company_profile_vectorizer.fit_transform(all_company_profiles_list)

		self.description_vectorizer = TfidfVectorizer(max_features=LARGE_TEXT_VECTOR_SIZE)
		all_descriptions_list = [self.preprocess_text(job_posting.description) for job_posting in self.job_postings_list]
		self.description_vectorizer.fit_transform(all_descriptions_list)

		self.requirement_vectorizer = TfidfVectorizer(max_features=LARGE_TEXT_VECTOR_SIZE)
		all_requirements_list = [self.preprocess_text(job_posting.requirement) for job_posting in self.job_postings_list]
		self.requirement_vectorizer.fit_transform(all_requirements_list)

		self.benefit_vectorizer = TfidfVectorizer(max_features=LARGE_TEXT_VECTOR_SIZE)
		all_benefits_list = [self.preprocess_text(job_posting.benefit) for job_posting in self.job_postings_list]
		self.benefit_vectorizer.fit_transform(all_benefits_list)

		self.employment_type_vectorizer = TfidfVectorizer(max_features=SMALL_TEXT_VECTOR_SIZE)
		all_employment_types_list = [self.preprocess_text(job_posting.employment_type) for job_posting in self.job_postings_list]
		self.employment_type_vectorizer.fit_transform(all_employment_types_list)

		self.required_experience_vectorizer = TfidfVectorizer(max_features=SMALL_TEXT_VECTOR_SIZE)
		all_required_experiences_list = [self.preprocess_text(job_posting.required_experience) for job_posting in self.job_postings_list]
		self.required_experience_vectorizer.fit_transform(all_required_experiences_list)

		self.required_education_vectorizer = TfidfVectorizer(max_features=SMALL_TEXT_VECTOR_SIZE)
		all_required_educations_list = [self.preprocess_text(job_posting.required_education) for job_posting in self.job_postings_list]
		self.required_education_vectorizer.fit_transform(all_required_educations_list)

		self.industry_vectorizer = TfidfVectorizer(max_features=MEDIUM_TEXT_VECTOR_SIZE)
		all_industries_list = [self.preprocess_text(job_posting.industry) for job_posting in self.job_postings_list]
		self.industry_vectorizer.fit_transform(all_industries_list)

		self.function_vectorizer = TfidfVectorizer(max_features=MEDIUM_TEXT_VECTOR_SIZE)
		all_functions_list = [self.preprocess_text(job_posting.function) for job_posting in self.job_postings_list]
		self.function_vectorizer.fit_transform(all_functions_list)

	@staticmethod
	def preprocess_text(text):
		text = re.sub("<[^>]*>", "", text)
		symbols = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
		text = (re.sub("[\W]+", " ", text.lower()) + " ".join(symbols).replace("-", ""))
		return text

	def __getitem__(self, index):
		if index in self.vectorized_job_postings_dict:
			return self.vectorized_job_postings_dict[index]
		else:
			vectorized_job_posting = deepcopy(self.job_postings_list[index])
			vectorized_job_posting.title = self.title_vectorizer.transform([vectorized_job_posting.title]).toarray()[0]
			vectorized_job_posting.location = self.location_vectorizer.transform([vectorized_job_posting.location]).toarray()[0]
			vectorized_job_posting.department = self.department_vectorizer.transform([vectorized_job_posting.department]).toarray()[0]
			vectorized_job_posting.company_profile = self.company_profile_vectorizer.transform([vectorized_job_posting.company_profile]).toarray()[0]
			vectorized_job_posting.description = self.description_vectorizer.transform([vectorized_job_posting.description]).toarray()[0]
			vectorized_job_posting.requirement = self.requirement_vectorizer.transform([vectorized_job_posting.requirement]).toarray()[0]
			vectorized_job_posting.benefit = self.benefit_vectorizer.transform([vectorized_job_posting.benefit]).toarray()[0]
			vectorized_job_posting.employment_type = self.employment_type_vectorizer.transform([vectorized_job_posting.employment_type]).toarray()[0]
			vectorized_job_posting.required_experience = self.required_experience_vectorizer.transform([vectorized_job_posting.required_experience]).toarray()[0]
			vectorized_job_posting.required_education = self.required_education_vectorizer.transform([vectorized_job_posting.required_education]).toarray()[0]
			vectorized_job_posting.industry = self.industry_vectorizer.transform([vectorized_job_posting.industry]).toarray()[0]
			vectorized_job_posting.function = self.function_vectorizer.transform([vectorized_job_posting.function]).toarray()[0]
			if len(list(vectorized_job_posting.salary_range[0].split("-"))) == 2:
				try:
					vectorized_job_posting.salary_range = tuple(map(int, vectorized_job_posting.salary_range[0].split("-")))
				except:
					vectorized_job_posting.salary_range = (0, 0)
			else:
				vectorized_job_posting.salary_range = (0, 0)
			vectorized_job_posting.telecommuting = int(vectorized_job_posting.telecommuting)
			vectorized_job_posting.has_company_logo = int(vectorized_job_posting.has_company_logo)
			vectorized_job_posting.has_question = int(vectorized_job_posting.has_question)
			vectorized_job_posting.fraudulent = int(vectorized_job_posting.fraudulent)
			self.vectorized_job_postings_dict[index] = vectorized_job_posting
			return self.vectorized_job_postings_dict[index]
