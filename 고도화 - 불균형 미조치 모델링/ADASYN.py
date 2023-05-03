#ADASYN 적용
from imblearn.over_sampling import *
adasyn = ADASYN(random_state = 1004)
X_train_over2, y_train_over2 = adasyn.fit_resample(X_train, y_train)
print('ADASYN 적용 전 학습용 피처/레이블 데이터 세트: ', X_train.shape, y_train.shape)
print('ADASYN 적용 전 레이블 값 분포: ')
print(pd.Series(y_train).value_counts())
print('\n')
print('ADASYN 적용 후 학습용 피처/레이블 데이터 세트: ', X_train_over2.shape, y_train_over2.shape)
print('ADASYN 적용 후 레이블 값 분포: ')
print(pd.Series(y_train_over2).value_counts())
print('\n')


#모델[1] KNN-ADASYN
from sklearn.neighbors import KNeighborsClassifier
#모형 객체 생성
knn_model = KNeighborsClassifier(n_neighbors = 5)
#train data를 가지고 모형 학습
knn_model.fit(X_train_over2, y_train_over2)
#test data를 가지고 y_hat을 예측(분류)
y_knn_hat = knn_model.predict(X_test)
#모형 성능 평가 - Confusion Matrix 계산
from sklearn import metrics
knn_matrix = metrics.confusion_matrix(y_test, y_knn_hat)
print(knn_matrix)
print('\n')
#모형 성능 평가 - 평가 지표 계산
knn_report = metrics.classification_report(y_test, y_knn_hat)
print(knn_report)


#모델[2] SVM - ADASYN
from sklearn import svm
#모형 객체 생성
svm_model = svm.SVC(kernel = 'rbf')
#train data를 가지고 모형 학습
svm_model.fit(X_train_over2, y_train_over2)
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


#모델[3] Decision Tree - ADASYN
from sklearn import tree
#모형 객체 생성
tree_model = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)
#train dat를 가지고 모형 학습
tree_model.fit(X_train_over2, y_train_over2)
#test data를 가지고 y_hat을 예측(분류)
y_tree_hat = tree_model.predict(X_test)
#모형 성능 평가 - Confusion Matrix 계산
from sklearn import metrics
tree_matrix = metrics.confusion_matrix(y_test, y_tree_hat)
print(tree_matrix)
print('\n')
#모형 성능 평가 - 평가 지표 계산
tree_report = metrics.classification_report(y_test, y_tree_hat)
print(tree_report)


#모델[4] Random Forest - ADASYN
from sklearn.ensemble import RandomForestClassifier
#모형 객체 생성
rf_model = RandomForestClassifier(n_estimators = 100)
#train data를 가지고 모형 학습
rf_model.fit(X_train_over2, y_train_over2)
#test data를 가지고 y_hat을 예측(분류)
y_rf_hat = rf_model.predict(X_test)
#모형 성능 평가 - Confusion Matrix 계산
from sklearn import metrics
rf_matrix = metrics.confusion_matrix(y_test, y_rf_hat)
print(rf_matrix)
print('\n')
#모형 성능 평가 - 평가 지표 계산
rf_report = metrics.classification_report(y_test, y_rf_hat)
print(rf_report)

