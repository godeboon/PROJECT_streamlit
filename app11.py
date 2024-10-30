import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 임베딩 모델 로드
encoder = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 식당 관련 질문과 답변 데이터
questions = [
     "알림의 목소리 크기를 어느 정도로 할까요?",
    "객체 검출의 확률이 어느 정도일 때 알림을 울리게 설정할까요 ?",
    "프로젝트 주제가 어떻게 되나요"
]

answers = [
 "3으로 해주세요",
    "50 % 일때 ",
    "통행약자들의 보행을 어씨스트 하는 기능을 구현해 서비스로 출품하는 것이 프로젝트의 주제입니다"
]

# 질문 임베딩과 답변 데이터프레임 생성
question_embeddings = encoder.encode(questions)
df = pd.DataFrame({'question': questions, '챗봇': answers, 'embedding': list(question_embeddings)})

# 대화 이력을 저장하기 위한 Streamlit 상태 설정
if 'history' not in st.session_state:
    st.session_state.history = []

# 챗봇 함수 정의
def get_response(user_input):
    # 사용자 입력 임베딩
    embedding = encoder.encode(user_input)
    
    # 유사도 계산하여 가장 유사한 응답 찾기
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    # 대화 이력에 추가
    st.session_state.history.append({"user": user_input, "bot": answer['챗봇']})

# Streamlit 인터페이스
st.title("통행약자 인도 보행 어씨스턴트 서비스")
st.write("이 서비스 관한 질문을 하세요 예: 프로젝트 주제가 어떻게 되나요? ")

user_input = st.text_input("user", "")

if st.button("Submit"):
    if user_input:
        get_response(user_input)
        user_input = ""  # 입력 초기화

# 대화 이력 표시
for message in st.session_state.history:
    st.write(f"**사용자**: {message['user']}")
    st.write(f"**챗봇**: {message['bot']}")