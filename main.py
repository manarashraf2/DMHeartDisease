import pandas as pd
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

HD=pd.read_csv(r"C:\Users\manar\Downloads\Compressed\Arabic tweets\heart.csv")



#check for duplicates and removing them
#HD = HD[HD.duplicated()]
#print(HD.shape)
HD=HD.drop_duplicates()
#HD = HD[HD.duplicated()]
#print(HD.shape)

#check for null values
#print(HD.isnull().sum())

#check for outliers
#plt.boxplot(HD.drop('target', axis=1))
#plt.show()

#remove outliers(Interquartile Range (IQR))
Q1 = HD.quantile(0.25)
Q3 = HD.quantile(0.75)
IQR = Q3-Q1
HD = HD[~((HD<(Q1-1.5*IQR))|(HD>(Q3+1.5*IQR))).any(axis=1)]

#check for outliers after removing them
#plt.boxplot(HD.drop('target', axis=1))
#plt.show()


#visualize the sex based prediction
pd.crosstab(HD.sex,HD.target).plot(kind="bar",figsize=(5,5),color=['blue','red' ])
plt.xlabel("Sex (0 = female, 1= male)")
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()

#visualize the age based prediction
pd.crosstab(HD.age,HD.target).plot(kind="bar",figsize=(10,10),color=['blue','red' ])
plt.xlabel("age")
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


#dataframe
HD=pd.DataFrame(HD)

#train and test data
x=HD.drop('target',axis=1)
y=HD.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)


def naive_bayes():
    #applaying naive bayes
    GHD=GaussianNB()
    GHD.fit(x_train, y_train)
    y_prediction = GHD.predict(x_test)

    #accuracy(naive bayes)
    print(f"Accuracy Score: {accuracy_score(y_test, y_prediction) * 100:.2f}%")
    print(classification_report(y_test,y_prediction))


def knn():
    #applying knn
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    y_prediction = knn.predict(x_test)

    #accuracy(knn)
    print(f"Accuracy Score: {accuracy_score(y_test, y_prediction) * 100:.2f}%")
    print(classification_report(y_test,y_prediction))



def Decisiontree_Classifier():

    categorical_columns = []
    continous_columns = []
    for column in HD.columns:
        if len(HD[column].unique()) <= 10:
            categorical_columns.append(column)
        else:
            continous_columns.append(column)

    #print(categorical_columns)

    #create dummies values
    categorical_columns.remove('target')
    data_set = pd.get_dummies(HD, columns = categorical_columns)

    #using standardScaler for features which .get_dummies can't process in it 'continous cloumns.
    st_scaler= StandardScaler()
    scale_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    data_set[scale_columns] = st_scaler.fit_transform(data_set[scale_columns])

    # applying decision tree
    tree_classifer = DecisionTreeClassifier(random_state=0)
    tree_classifer.fit(x_train, y_train)
    pred = tree_classifer.predict(x_test)

    # accuracy(decision tree)
    print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
    print(classification_report(y_test, pred))


naive_bayes()
knn()
Decisiontree_Classifier()
