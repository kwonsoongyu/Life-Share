#군집화 진행 후 모델링
#(1) KNN (2) SVM (3) Decision Tree (4) random Forest
#22. 속성(변수) 선택
X = df_1[['기준년월', '시군구코드', '연령구간코드', '인구수', '월소득평균금액', '대출평균잔액', '평균신용평가
점수', '카드이용평균금액', '보유승용차평균가격']]
y = df_1['cluster_fn'] ##종속 변수 y
#설명 변수 데이터를 정규화
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
#train data와 test data로 구분(75:25)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =
1004)
print('train data 개수: ', X_train.shape) #train data 개수: (654, 11)
print('test data 개수: ', X_test.shape) #test data 개수: (219, 11)
          
#모델[1] KNN
from sklearn.neighbors import KNeighborsClassifier
#모형 객체 생성
knn_model = KNeighborsClassifier(n_neighbors = 5)
#train data를 가지고 모형 학습
knn_model.fit(X_train, y_train)
#test data를 가지고 y_hat을 예측(분류)
y_knn_hat = knn_model.predict(X_test)
#모형 성능 평가 - Confusion Matrix 계산
from sklearn import metrics
knn_matrix = metrics.confusion_matrix(y_test, y_knn_hat)
print(knn_matrix)
print('\n')
#모형 성능 평가 - 평가 지표 계산
knn_report = metrics.classification_report(y_test, y_knn_hat,)
print(knn_report)
          
# 모델[2] SVM
from sklearn import svm
#모형 객체 생성
svm_model = svm.SVC(kernel = 'rbf')
#train data를 가지고 모형 학습
svm_model.fit(X_train, y_train)
#test data를 가지고 y_hat을 예측(분류)
y_svm_hat = svm_model.predict(X_test)
#모형 성능 평가 - Confusion Matrix 계산
from sklearn import metrics
svm_matrix = metrics.confusion_matrix(y_test, y_svm_hat)
print(svm_matrix)
print('\n')
#모형 성능 평가 - 평가 지표 계산
svm_report = metrics.classification_report(y_test, y_svm_hat)
print(svm_report)
print('\n')
          
#모델 [3] Decision Tree
from sklearn import tree
#모형 객체 생성
tree_model = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)
#train data를 가지고 모형 학습
tree_model.fit(X_train, y_train)
#test data를 가지고 y_hat을 예측 (분류)
y_tree_hat = tree_model.predict(X_test)
#모평 성능 평가 - Confusion Matrix 계산
from sklearn import metrics
tree_matrix = metrics.confusion_matrix(y_test, y_tree_hat)
print(tree_matrix)
print('\n')
#모형 성능 평가 - 평가 지표 계산
tree_report = metrics.classification_report(y_test, y_tree_hat)
print(tree_report)
          
#모델 [4] Random Forest
from sklearn.ensemble import RandomForestClassifier
#모형 객체 생성
rf_model = RandomForestClassifier(n_estimators= 100)
#train data를 가지고 모형 학습
rf_model.fit(X_train, y_train)
#test data를 가지고 y_hat을 예측(분류)
y_rf_hat = rf_model.predict(X_test)
# 모형 성능 형가 - Confusion Matrix 계산
from sklearn import metrics
rf_matrix = metrics.confusion_matrix(y_test, y_rf_hat)
print(rf_matrix)
print('\n')
#모형 성능 평가 - 평가 지표 계산
rf_report = metrics.classification_report(y_test, y_rf_hat)
print(rf_report)          
          
          
          
          
          
          
