import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


#1. load data 
res_df = pd.read_csv("src\datasets\merged_resume2.csv")
req_df = pd.read_csv("src/datasets/merged_recruitment2.csv")
answer_df = pd.read_csv("src/datasets/apply_train.csv")

#2. drop columns
resume_df = res_df.drop(columns=['univ_transfer', 'univ_location', 'hischool_type_seq', 'hischool_location_seq','job_code_seq2','job_code_seq3'], inplace=False)
recruitment_df = req_df.drop(columns=['career_start','career_end'], inplace=False)

recruitment_df['address_seq1'] = recruitment_df['address_seq1'].astype(str).replace('nan', '')
recruitment_df['address_seq2'] = recruitment_df['address_seq2'].astype(str).replace('nan', '')
recruitment_df['address_seq3'] = recruitment_df['address_seq3'].astype(str).replace('nan', '')

# 변환된 컬럼들을 합쳐 'adress_merged' 컬럼 생성
recruitment_df['adress_merged'] = recruitment_df['address_seq1'] + recruitment_df['address_seq2'] + recruitment_df['address_seq3']

# 커리어 시작과 끝으로 carrer_duration 컬럼생성
# recruitment_df['career_duration'] = recruitment_df['career_end'] - recruitment_df['career_start']
# recruitment_df.drop(columns = ['career_end', 'career_start'], inplace = True)

# text preprocess
resume_texts = resume_df[['reg_date','updated_date','text_keyword','job_code_seq1','career_job_code','certificate_contents','hischool_special_type','hischool_nation','hischool_gender','univ_major','univ_sub_major']].fillna('')  # 결측치는 빈 문자열로 대체
recruitment_texts = recruitment_df[['check_box_keyword','text_keyword']].fillna('')

resume_categories = resume_df[['degree', 'univ_type_seq1', 'univ_type_seq2', 'univ_major_type', 'language', 'exam_name']] 
recruitment_categories = recruitment_df[['company_type_seq', 'supply_kind', 'address_merged', 'education', 'major_task', 'qualifications']]

resume_mode = resume_categories.mode().iloc[0]
recruitment_mode = recruitment_categories.mode().iloc[0]
resume_categories_filled = resume_categories.fillna(resume_mode)
recruitment_categories_filled = recruitment_categories.fillna(recruitment_mode)

resume_nums = resume_df[['graduate_date', 'hope_salary', 'last_salary', 'carrer_month', 'univ_score', 'score']]
recruitment_nums =  recruitment_df[['employee']]
# 모든 데이터 타입를 하나의 리스트로 합침
all_texts = pd.concat([resume_texts, recruitment_texts], ignore_index=True)

all_categories = pd.concat([resume_categories, recruitment_categories], ignore_index=True)

#categories 라벨 인코딩
all_categories.apply(LabelEncoder().fit_transform)
all_categories

# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)