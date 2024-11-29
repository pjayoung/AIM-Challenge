# AIM-Challenge (2024 CO-Show 및 AIM 챌린지, 인공지능 모델 개발 대회)
---
<br/>

## 1. 배경 & 목적
- 대회개요 : 더욱 편리하고 사용자 친화적인 키오스크 환경을 조성하기 위해, 멀티모달 AI 모델을 개발하고 키오스크 시스템에 적용하는 것을 목표로 함.
- 대회주제 : 배리어프리 멀티모달 모델
- 대회목적 : AI 분야 실무 경험 제공, 창의적 문제해결 능력 강화, 우수 인재 발굴 및 산학 연계 강화
<br/>
<br/>

## 2. 주최/주관 & 참가 대상 & 성과
- 주최: 경북대학교 인공지능 혁신융합대학사업단
- 참가대상 : 서울시립대학교 학부 재학생 중 인공지능 분야 관련 교과목 이수자, 프로그래밍 능력 보유자 (그 외 성균관대, 서울과학기술대 등 다수 인공지능 혁신융합대학사업단 가입 학교 참여)
- 평가지표 : 모델 성능 (70%) + 발표 (30%) (내부 심사 기준에 따름)
- 성과: 기술상
<br/>
<br/>

## 3. 프로젝트 기간
- 프로젝트 기간
  1. 부트캠프 : 2024.08.26(월) - 08.28(수)
  2. 예선 멘토링 : 2024.9.02(월 - 09.27(금)
  4. 본선 멘토링 : 2024.09.30(월) - 11.15(금)
- 코드 및 발표자료 제출
  1. 예선 : 2024.09.27(금) 13:00-17:00
  2. 본선 및 결선 : 11.20(수) - 11.22(금)
- 최종 수상자 발표일 : 11.22(금) 최종 평가 및 시상
<br/>
<br/>

## 4. 내용
- (광명테크) 배리어프리 키오스크의 현재 문제점을 해결하는 모델 개발
- 키오스크에 올리 수 있는 API 형태로 제작
- 이때, 키오스크는 CPU 환경임을 감안 (제한된 환경)
<br/>
<br/>

## 5. 담당 역할 (4인 4모델 개발)
- 정하연(팀장) : STT 모델 개발
- 박자영 : FACE & Age Classification 모델 개발
- 서가원 : Eye Tracking 모델 개발
- 여시형 : Hand Gesture 모델 개발
<br/>
<br/>

## 6. 프로젝트 구성
### 베리어프리 키오스크란?
- 장애인, 고령자 등 모든 사용자가 편리하게 이용할 수 있도록 설계된 무인 정보 단말기
- 본 대회에서는 얼굴 인식, 아이트레킹, 핸드 제스처, 음성 인식이 탑재된 키오스크 제작
<br/>

### FACE & Age Classification 모델 방법론
- 데이터 소개 :
  1. UTKFace (백인 비율이 많은 얼굴 이미지 데이터셋으로 얼굴, 나이, 인종, 성별 등에 대한 정보 존재)
  2. All Age Faces (아시안 비율이 많은 얼굴 이미지 데이터셋으로 얼굴, 나이, 성별에 대한 정보 존재)
- 모델 아키텍쳐 :
  1. 얼굴 탐지 - OpenCV DNN 기반 모델 활용
  2. 나이 분류 - CNN 구조를 활용한 논문 참고 (Ordinal Regression with Multiple Output CNN)
- 손실 함수 : Cross-Entropy Loss
- 평가 지표 : MAE (회귀 기반 코드 활용)
<br/>

### 나이 분류 모델 학습
- 데이터 전처리 : 크기 조정 및 train 데이터 Augmentation
- 모델 학습 과정 : (CNN) 나이 입력값 범주화 -> 다중 출력 방식 활용, 어느 범주에 속하는지 이진 분류로 판단 -> 범주별 중요도 활용, Cross-Entropy Loss 데이터 불균형 완화 -> 나이 순서 관계 보존 가능!
<div align="center">
<img width="622" alt="스크린샷 2024-11-29 오전 11 38 07" src="https://github.com/user-attachments/assets/88fa7b69-bb0a-4c56-b762-379177dbcad1">
</div>
<br/>

### 분석 결과 및 결론
- 실험 결과
  1. MAE : 0.865
  2. 처리 속도 : (얼굴 인식) 35-50 msec / (나이 분류) 35-40 msec
- 의의
  1. CPU라는 제한된 환경에서 Age Classification 도전 및 기존 코드에 비해 성능 향상 성공
  2. GPU 서버 사용법 숙달
- 한계 및 보완점
  1. 기존에 비해 성능이 올라가긴 했으나 아직 미흡한 수준, 더 많은 데이터셋과 다양한 기법 필요
  2. 제한된 환경에서 높은 성능을 내기 위한 "한 방"을 찾아내야 함.
<br/>
<br/>


## 7. 증빙자료
- 코드
  1. 0. face_live_API.py (웹캠)
  2. 0. test_API.py (이미지)
- 데이터
  1. Total : UTKFace + AAF (파일이 커서 업로드 불가능)
  2. 12575A57.jpg (AAF 중 예시 이미지)
- PPT : [구글 슬라이드 자료](https://docs.google.com/presentation/d/1MJ8HSAJ4GOyx-jy3035qhbwsTHnRBNjQm8rXLN2LseI/edit?usp=sharing)

## 8. 참고문헌
[얼굴 인식](https://github.com/smahesh29/Gender-and-Age-Detection)
[나이 예측, Ordinal_Regression_with_Multiple_Output_CNN_for_Age_Estimation](https://github.com/xjtulyc/Ordinal_Regression_with_Multiple_Output_CNN_for_Age_Estimation/tree/main/dataset)
