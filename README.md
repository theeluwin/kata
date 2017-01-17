# 수우련을 하자

[@theeluwin](https://twitter.com/theeluwin)'s personal code kata.

"HMM은 기본적이고 고전적인 알고리즘이니까"라고 말하고 다니기엔 '그럼 나는 저걸 짤 수나 있나?' 싶어서 그걸 확인하고 기록해두기 위함입니다.

알고리즘 구현 능력을 마치 뜨거운 모래에 손가락 찌르기를 하듯 하드보일드하게 수련을 해야할듯 하여.. 관련 글 - [당신이 제자리 걸음인 이유: 지루하거나 불안하거나](http://egloos.zum.com/agile/v/5749946)

주로 머신러닝 관련 알고리즘과 방법론을 구현하며, 딥러닝의 경우는 기본적인 부분 빼고는 [Keras](https://keras.io/)나 [Tensorflow](https://www.tensorflow.org/)를 사용 할 예정.

`tasks`엔 머신러닝 관련을, `algorithms`엔 [코세라 스탠포드 알고리즘 강의](https://www.coursera.org/learn/algorithms-divide-conquer)에서 다룬 알고리즘들을 구현함.

머신러닝의 경우 코딩의 [제약 조건](https://namu.wiki/w/%EC%A0%9C%EC%95%BD%28%ED%97%8C%ED%84%B0X%ED%97%8C%ED%84%B0%29)이 다음과 같이 이음:

1. 파이썬3, PEP8에서 E501 빼고
2. 주석은 안써도 되지만 코드 퀄리티는 남들에게 보여줄 수 있을 정도는 되어야 함
3. [numpy](http://www.numpy.org/), [scikit-learn](http://scikit-learn.org/stable/) 등에서 제공되는 보조도구까진 사용해도 되지만 알고리즘 로직 자체는 직접 짜야함
4. 그냥 짤 수 있는것도 굳이 [numpy](http://www.numpy.org/)를 활용해서 해볼 것
5. 알고리즘이 실제로 쓰이는 task가 있어야 함
	* 이를 해결하고 evaluation하는 부분까지 구현
	* 외부 라이브러리에서 제공 되거나 간단하게 구현해 볼 수 있는 다른 알고리즘도 같이 포함 시켜서 비교 해보기 (`baseline`)
6. 데이터셋 포함
	* 크기가 너무 크면 다운로더를 제공
	* 공개된 자료(출처 명시) 혹은 쉽게 생성해낼 수 있는 데이터를 사용
7. 라이브러리 수준으로 최대한 범용적이게 구현할것
8. `utils/skeleton`에 있는 형식을 사용할것
    * 이렇게 싱글톤으로 스크립트를 짜게되면 어디까지를 공통분모로 처리할지가 애매해지는데, 기준은 각각의 메소드가 유닛테스트를 하기에 적합하도록 쪼개기
    * 따라서 적합한 유닛테스트 작성도 이 수련에 포함됨
9. 사실 위 1~8 원칙을 열심히 지키려고 했으나 어느새 망가져서 그냥 코드 정리도 포기하고 파이썬2, 3도 섞여있고 막 그럼ㅎㅎ.. 에라 모르겠다. 그냥 한번씩 구현 해봤다는것을 온전히 기록하기 위함일 뿐입니다. 흥흥흥~

Bonus Point:
* test set까지 학습해버리면(cheating) 정말로 성능이 증가하는지 체크
* partial fit이 가능하면 구현 해볼것
* 예쁘게 figure 그리기

<br />

## Usage

충분히 범용적으로 쓰일 수 있도록 코딩해둔 모듈들:

- [HMM]()
- [Naive Bayes Classifier]()
- [Collaborative Filtering]()
- [Linear Regression]()

테스트는 각 task 디렉토리 내에서 `test.py`를 `-m` 옵션 없이 실행시키면 됨!

`test.py benchmark`로 실행 시키면 유닛테스트 대신에 모든 메소드를 싹 돌려서 성능 비교표를 뽑아줌

(이렇게 되도록 하는 작업은 아직 하는중)

<br />

## Todo

Task와 그걸 해결 할 수 있는 알고리즘들의 나열로 기재. "hard-coded"라고 써있는 부분이 이 프로젝트의 핵심.

#### [POS Tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging)

- [x] [CRF](https://en.wikipedia.org/wiki/Conditional_random_field) with [pycrfsuite](https://python-crfsuite.readthedocs.io/en/latest/) `baseline`
- [x] [HMM](https://en.wikipedia.org/wiki/Hidden_Markov_model), hard-coded

#### Classification: [Titanic Survival](https://www.kaggle.com/c/titanic)

- [x] [Random Forest Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) from [sklearn](http://scikit-learn.org/) `baseline`
- [x] [Naive Bayes Classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier), hard-coded

#### Recommendation: [MovieLens](http://grouplens.org/datasets/movielens/)

- [x] [Model-Based CF](https://en.wikipedia.org/wiki/Collaborative_filtering#Model-based) with [scipy](http://www.scipy.org/) `baseline`
- [x] [Memory-Based CF](https://en.wikipedia.org/wiki/Collaborative_filtering#Memory-based), hard-coded

#### [Polynomial Regression](https://en.wikipedia.org/wiki/Polynomial_regression)

- [x] [numpy polyfit](http://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html) `baseline`
- [x] Linear Regression
	- [x] using TF
	- [x] hard-coded
- [x] Normal Equation

#### Classification: [Iris Flower](https://en.wikipedia.org/wiki/Iris_flower_data_set)

- [ ] [FFNN](https://en.wikipedia.org/wiki/Feedforward_neural_network)
	- [x] using Keras `baseline`
	- [ ] using TF
	- [ ] hard-coded
- [ ] [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)
	- [ ] using TF
	- [ ] hard-coded
- [ ] [SVM](https://en.wikipedia.org/wiki/Support_vector_machine), hard-coded
- [ ] [Random Forest](https://en.wikipedia.org/wiki/Random_forest), hard-coded

#### Clustering: Random Colony

- [ ] [k-means](https://en.wikipedia.org/wiki/K-means_clustering), hard-coded
- [ ] [Hierarchical](https://en.wikipedia.org/wiki/Hierarchical_clustering), hard-coded
- [ ] [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN), hard-coded

#### [Word Embedding](https://en.wikipedia.org/wiki/Word_embedding)

- [x] [GloVe](http://www.aclweb.org/anthology/D14-1162), hard-coded
- [x] [Word2Vec](https://en.wikipedia.org/wiki/Word2vec), hard-coded

#### Classification: [MNIST](http://yann.lecun.com/exdb/mnist/)

- [ ] [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)
	- [ ] using Keras `baseline`
	- [ ] using TF

#### [Language Model](https://en.wikipedia.org/wiki/Language_model)

- [ ] [NPLM](http://www.jmlr.org/papers/v3/bengio03a.html), using TF `baseline`
- [ ] [RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network)
    - [ ] using Keras
    - [ ] using TF
    - [ ] [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory), using TF

이정도면 아마 고전적인 머신러닝 알고리즘들은 대부분 커버할 수 있을듯?

<br />

## Advanced

최신 논문들을 TF로 구현해보기.

- [ ] [Effective Approaches to Attention-based Neural Machine Translation](http://arxiv.org/abs/1508.04025)
- [ ] [LCSTS: A Large Scale Chinese Short Text Summarization Dataset](http://arxiv.org/abs/1506.05865)
- [ ] [Towards Abstraction from Extraction: Multiple Timescale Gated Recurrent Unit for Summarization](http://arxiv.org/abs/1607.00718)
