import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import surprise
from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict


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


# 시각화
recruitment_df.columns
resume_df.columns

# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)




# Surprise 데이터셋 생성
interaction_df = resume_df[['resume_seq', 'recruitment_seq']].drop_duplicates()
interaction_df['rating'] = 1  # 지원 여부를 나타내는 평가 점수로 1을 할당



# 'surprise'에 필요한 데이터 형식으로 변환
reader = Reader(rating_scale=(1, 1))  # 모든 평가 점수가 1이므로
data = Dataset.load_from_df(interaction_df[['resume_seq', 'recruitment_seq', 'rating']], reader)

# 데이터를 훈련 세트와 테스트 세트로 분할
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# SVD 모델 훈련
svd_model = SVD(random_state=42)
svd_model.fit(trainset)

# 모든 이력서에 대한 모든 채용공고의 추천 점수를 예측합니다.
def get_top_n(predictions, n=5):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # 각 사용자별로 상위 N개의 추천을 정렬합니다.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

all_recruitment = recruitment_df['recruitment_seq'].unique()
predictions = []
for resume_seq in resume_df['resume_seq'].unique():
    resume_already_applied = resume_df[resume_df['resume_seq'] == resume_seq]['recruitment_seq'].tolist()
    resume_not_applied = np.setdiff1d(all_recruitment, resume_already_applied)
    predictions.extend([svd_model.predict(resume_seq, recruitment_seq) for recruitment_seq in resume_not_applied])

top_n = get_top_n(predictions, n=5)

# 추천 결과를 적절한 형식으로 저장합니다.
submission_df = pd.DataFrame([(uid, [iid for (iid, _) in user_ratings]) for uid, user_ratings in top_n.items()], columns=['resume_seq', 'recruitment_seq'])
submission_df = submission_df.explode('recruitment_seq')

# recall5 평가 메트릭
def recall5(answer_df, submission_df):
    """
    Calculate recall@5 for given dataframes.
    
    Parameters:
    - answer_df: DataFrame containing the ground truth
    - submission_df: DataFrame containing the predictions
    
    Returns:
    - recall: Recall@5 value
    """
    
    primary_col = answer_df.columns[0]
    secondary_col = answer_df.columns[1]
    
    # Check if each primary_col entry has exactly 5 secondary_col predictions
    prediction_counts = submission_df.groupby(primary_col).size()
    if not all(prediction_counts == 5):
        raise ValueError(f"Each {primary_col} should have exactly 5 {secondary_col} predictions.")


    # Check for NULL values in the predicted secondary_col
    if submission_df[secondary_col].isnull().any():
        raise ValueError(f"Predicted {secondary_col} contains NULL values.")
    
    # Check for duplicates in the predicted secondary_col for each primary_col
    duplicated_preds = submission_df.groupby(primary_col).apply(lambda x: x[secondary_col].duplicated().any())
    if duplicated_preds.any():
        raise ValueError(f"Predicted {secondary_col} contains duplicates for some {primary_col}.")


    # Filter the submission dataframe based on the primary_col present in the answer dataframe
    submission_df = submission_df[submission_df[primary_col].isin(answer_df[primary_col])]
    
    # For each primary_col, get the top 5 predicted secondary_col values
    top_5_preds = submission_df.groupby(primary_col).apply(lambda x: x[secondary_col].head(5).tolist()).to_dict()
    
    # Convert the answer_df to a dictionary for easier lookup
    true_dict = answer_df.groupby(primary_col).apply(lambda x: x[secondary_col].tolist()).to_dict()
    
    
    individual_recalls = []
    for key, val in true_dict.items():
        if key in top_5_preds:
            correct_matches = len(set(true_dict[key]) & set(top_5_preds[key]))
            individual_recall = correct_matches / min(len(val), 5) # 공정한 평가를 가능하게 위하여 분모(k)를 'min(len(val), 5)' 로 설정함 
            individual_recalls.append(individual_recall)


    recall = np.mean(individual_recalls)
    
    return recall

# Recall@5를 계산합니다.
recall_at_5 = recall5(answer_df, submission_df)
print(f"Recall@5: {recall_at_5}")

# Sorting the submission based on 'resume_seq' in ascending order
submission_df = submission_df.sort_values(by='resume_seq')

# Save the sorted recommendations to a CSV file
submission_df.to_csv('colab_submission1.csv', index=False)