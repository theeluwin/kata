## 수우련을 하자

[@theeluwin](https://twitter.com/theeluwin)'s personal code kata.

"HMM은 기본적이고 고전적인 알고리즘이니까"라고 말하고 다니기엔 '그럼 나는 저걸 짤 수나 있나?' 싶어서 그걸 확인하고 기록해두기 위함입니다.

알고리즘 구현 능력을 마치 뜨거운 모래에 손가락 찌르기를 하듯 하드보일드하게 수련을 해야할듯 하여.. 관련 글 - [당신이 제자리 걸음인 이유: 지루하거나 불안하거나](http://egloos.zum.com/agile/v/5749946)

주로 머신러닝 관련 알고리즘과 방법론을 구현하며, 딥러닝의 경우는 기본적인 부분 빼고는 [텐서플로우](https://www.tensorflow.org/)를 사용 할 예정.

코딩의 [제약 조건](https://namu.wiki/w/%EC%A0%9C%EC%95%BD%28%ED%97%8C%ED%84%B0X%ED%97%8C%ED%84%B0%29)으로는 다음이 있음:

1. 파이썬 2, 3 모두 지원되는 코드
	* 나눠서 짜는게 아니라 [코드 자체에 compatibility](http://python-future.org/compatible_idioms.html)가 있도록
2. 주석은 안써도 되지만 코드 퀄리티는 남들에게 보여줄 수 있을 정도는 되어야 함
3. [numpy](http://www.numpy.org/), [scikit-learn](http://scikit-learn.org/stable/) 등에서 제공되는 보조도구까진 사용해도 되지만 알고리즘 로직 자체는 직접 짜야함
4. 그냥 짤 수 있는것도 굳이 [numpy](http://www.numpy.org/)를 활용해서 해볼 것
5. 알고리즘이 실제로 쓰이는 하나의 task가 있어야 함
	* 이를 해결하고 evaluation하는 부분까지 구현
	* 외부 라이브러리에서 제공되는, 해당 task를 해결 할 수 있는 다른 알고리즘도 같이 포함 시켜서 비교 해보기
6. 데이터셋 포함
	* 크기가 너무 크면 다운로더를 제공
	* 공개된 자료(출처 명시) 혹은 쉽게 생성해낼 수 있는 데이터를 사용
7. 최대한 범용적이게 구현할것 (예시: [HMM 구현체](https://github.com/theeluwin/kata/blob/master/machine_learning/hmm/hmm.py), [Naive Bayes 구현체](https://github.com/theeluwin/kata/blob/master/machine_learning/naive_bayes/naive_bayes.py))

Bonus Point:
* partial fit이 가능하면 구현 해볼것!
* test set까지 학습해버리면(cheating) 정말로 성능이 증가하는지 체크
* figure 이쁘게 그리기

<br>

## Todo

"구현할 알고리즘 / 비교용 알고리즘 / 풀만한 task" 형식으로 기재

- [X] Linear Regression - Gradient Descent, Exact Solution / Polyfit by NP, LR with TF / Polynomial Fitting
- [x] Collaborative Filtering (Memory-Based) / Collaborative Filtering (Model-Based) / [MovieLens](http://grouplens.org/datasets/movielens/)
- [x] [HMM](https://en.wikipedia.org/wiki/Hidden_Markov_model) / [CRF](https://en.wikipedia.org/wiki/Conditional_random_field) / [POS tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging)
- [x] [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) / [Random Forest](https://en.wikipedia.org/wiki/Random_forest) / [Titanic Survival](https://www.kaggle.com/c/titanic)
- [ ] [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression), [FFNN](https://en.wikipedia.org/wiki/Feedforward_neural_network) / FFNN with TF / [Iris Flower](https://en.wikipedia.org/wiki/Iris_flower_data_set)
- [SVM](https://en.wikipedia.org/wiki/Support_vector_machine) / ? / ?
- [Random Forest](https://en.wikipedia.org/wiki/Random_forest) / ? / ?

(쓰다가 귀찮아짐)

k-means, agglomerative clustering, dbscan, svm.. 등등도 있고 뉴럴넷은 가장 기본적인 형태의 FFNN, CNN, RNN 까지만 텐서플로우 없이 구현 해보는걸 목표로.. 좀 무리수인가? 여기까지도 이미 솔직히 버겁다 인생
