#import the modules for SVM classification
from sklearn import svm as svm
from sklearn import datasets as datasets
#import the module for returning a plot
import matplotlib.pyplot as plt
#import the module for reading the data file
import csv
#import module to stop convergence warning from LinearSVC.fit()
import warnings as warn

#define function to load data file and save it as a dataset list
def loadDataset(file):
    with open(file,'r') as dataInput:
        data = csv.reader(dataInput)
        dataset = list(data)
        
        return dataset

#define function to setup five crossfold validation 
#as well as 20% of testSet as a val set for hyperparameters
def setFolds(data,i,trainingSet,testSet,valSet):    
    valcount = 0
    
    for j in range(len(data)):
        if j % 5 == i:
            if valcount == 0:
                valcount = 4
                valSet.append(data[j])
            else:
                valcount -= 1
                testSet.append(data[j])
        else:
            trainingSet.append(data[j])

#load the data from the file    
data = loadDataset('../Data/glass.data')

#setup total calculations for SVM kernels
accTotalRBF = 0
accTotalLin = 0
accTotalPoly = 0
accTotalSig = 0

#setup total calculations for One v.s. All
accTotalOVA = 0

#setup total calculations for SVM kernels with class weights
accTotalRBFW = 0
accTotalLinW = 0
accTotalPolyW = 0
accTotalSigW = 0

#setup the weight by comparing them to class most represented and adjusting accordingly
w = {'1': 76/70*100, '2': 1*100, '3': 76/17*100, '5': 76/13*100, '6': 76/9*100, '7': 76/29*100}

#Start each iteration of the five crossfold validation
for i in range(5):
    #initialize empty test, training and val sets as well as sets for just for their target labels
    testSet = []
    testSetT = []
    trainingSet = []
    trainingSetT = []        
    valSet = [] 
    valSetT = []
        
    #initialize hyperparameters sets along with their max value variable and max for the svm function    
    c = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    maxC = 0
    g = [3,2,1,0.1,0.01,0.001, 0.0001, 0.00001, 0.000001]    
    maxG = 0
    d = [2,3,4,5,6,7]
    maxD = 0
    coef = [0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 7, 10]
    maxCoef = 0
    currMax = 0
    
    #divide the data between the sets
    setFolds(data,i,trainingSet,testSet,valSet)
    
    #setup the label sets
    for i in range(len(trainingSet)):        
        trainingSetT.append(trainingSet[i][-1])        
    
    for i in range(len(testSet)):
        testSetT.append(testSet[i][-1])        
    
    for i in range(len(valSet)):
        valSetT.append(valSet[i][-1])
        
    
    #One V.S. One
    
    #RBF
    
    #Hyperparamater validation for c and gamma and return argMax
    currMax = 0
    for i in range(len(c)):
        for j in range(len(g)):
            predictSet = []
            accCount = 0        
            rbf = svm.SVC(kernel='rbf', C = c[i], gamma = g[j])
            rbf.fit(trainingSet,trainingSetT)
            predictSet = rbf.predict(valSet)
        
            for k in range(len(predictSet)):
                if(predictSet[k] == valSetT[k]):
                    accCount += 1
            if(accCount / len(valSet) * 100 > currMax):
                currMax = accCount / len(valSet) * 100
                maxC = i
                maxG = j

    #use argmax to predict data and add the accuracy to the total
    predictSet = []
    accCount = 0
    rbf = svm.SVC(kernel='rbf', C = c[maxC], gamma = g[maxG])
    rbf.fit(trainingSet,trainingSetT)    
    predictSet = rbf.predict(testSet)    
   
    for i in range(len(predictSet)):
        if(predictSet[i] == testSetT[i]):
            accCount += 1
            
    accTotalRBF += accCount / len(predictSet) * 100    

    
    #Linear
    
    #Hyperparamater validation for c and return argMax
    currMax = 0
    for i in range(len(c)):
        predictSet = []
        accCount = 0
        linear = svm.SVC(kernel ='linear', C = c[i])
        linear.fit(trainingSet,trainingSetT)
        predictSet = linear.predict(valSet)
        
        for j in range(len(predictSet)):
            if(predictSet[k] == valSetT[k]):
                accCount += 1
        if(accCount / len(valSet) * 100 > currMax):
            currMax = accCount / len(valSet) * 100
            maxC = i
        
    #use argmax to predict the data and add the accuracy to the total
    predictSet = []
    accCount = 0
    linear = svm.SVC(kernel='linear', C = c[maxC])
    linear.fit(trainingSet,trainingSetT)
    predictSet = linear.predict(testSet)    
   
    for i in range(len(predictSet)):
        if(predictSet[i] == testSetT[i]):
            accCount += 1
            
    accTotalLin += accCount / len(predictSet) * 100    

  
    #Polynomial
    
    #Hyperparamater validation for c, gamma and degree, and return argMax
    currMax = 0
    for i in range(len(c)):
        for j in range(len(g)):
            for k in range(len(d)):            
                predictSet = []
                accCount = 0
                polynomial = svm.SVC(kernel='poly', C = c[i], gamma = g[j], degree = d[k])
                polynomial.fit(trainingSet,trainingSetT)
                predictSet = polynomial.predict(valSet)
        
                for l in range(len(predictSet)):
                    if(predictSet[l] == valSetT[l]):
                        accCount += 1
                if(accCount / len(valSet) * 100 > currMax):
                    currMax = accCount / len(valSet) * 100                    
                    maxC = i
                    maxG = j
                    maxD = k
            
    #use argmax to predict data and add the accuracy to the total
    predictSet = []
    accCount = 0
    polynomial = svm.SVC(kernel='poly', C = c[maxC], gamma = g[maxG], degree = d[maxD])
    polynomial.fit(trainingSet,trainingSetT)
    predictSet = polynomial.predict(testSet)    
   
    for i in range(len(predictSet)):
        if(predictSet[i] == testSetT[i]):
            accCount += 1
            
    accTotalPoly += accCount / len(predictSet) * 100  
    
    
    #Sigmoid
    
    #Hyperparamater validation for c, gamma and coef, and return argMax  
    currMax = 0
    for i in range(len(c)):
        for j in range(len(g)):
            for k in range(len(coef)):
                predictSet = []
                accCount = 0
                sigmoid = svm.SVC(kernel='sigmoid', C = c[i], gamma = g[j], coef0 = coef[k])
                sigmoid.fit(trainingSet,trainingSetT)
                predictSet = sigmoid.predict(valSet)
        
                for l in range(len(predictSet)):
                    if(predictSet[l] == valSetT[l]):
                        accCount += 1
                if(accCount / len(valSet) * 100 > currMax):
                    currMax = accCount / len(valSet) * 100
                    maxC = i
                    maxG = j       
                    maxCoef = k
    
    #use argmax to predict data and add the accuracy to the total
    predictSet = []
    accCount = 0
    sigmoid = svm.SVC(kernel='sigmoid', C = c[maxC], gamma = g[maxG], coef0 = coef[maxCoef])
    sigmoid.fit(trainingSet,trainingSetT)
    predictSet = sigmoid.predict(testSet)    
   
    for i in range(len(predictSet)):
        if(predictSet[i] == testSetT[i]):
            accCount += 1
            
    accTotalSig += accCount / len(predictSet) * 100 
    


    #One V.S. All
    
    #suppress failed convergence warnings from fit()
    warn.filterwarnings('ignore','.*')
    
    #setup val and test sets as floats for LinearSVC    
    valSetFloat = [list(map(float,i)) for i in valSet]    
    testSetFloat = [list(map(float,i)) for i in testSet]
    
    #Hyperparamater validation for c and return argMax
    currMax = 0
    for i in range(len(c)):
        predictSet = []
        accCount = 0        
        ova = svm.LinearSVC(C = c[i], multi_class = 'ovr')
        ova.fit(trainingSet,trainingSetT)
        predictSet = ova.predict(valSetFloat)
    
        for j in range(len(predictSet)):
            if(predictSet[j] == valSetT[j]):
                accCount += 1
        if(accCount / len(valSet) * 100 > currMax):
            currMax = accCount / len(valSetFloat) * 100
            maxC = i            

    #use argmax to predict data and add the accuracy to the total
    predictSet = []
    accCount = 0
    ova = svm.LinearSVC(C = c[maxC], multi_class = 'ovr')
    ova.fit(trainingSet,trainingSetT)    
    predictSet = ova.predict(testSetFloat)    
   
    for i in range(len(predictSet)):
        if(predictSet[i] == testSetT[i]):
            accCount += 1
            
    accTotalOVA += accCount / len(predictSet) * 100


    
    #Weighted
    
    #RBF
    
    #Hyperparamater validation for c and gamma, and return argMax
    currMax = 0
    for i in range(len(c)):
        for j in range(len(g)):
            predictSet = []
            accCount = 0       
            rbf = svm.SVC(class_weight = w, kernel='rbf', C = c[i], gamma = g[j])
            rbf.fit(trainingSet,trainingSetT)
            predictSet = rbf.predict(valSet)
        
            for k in range(len(predictSet)):
                if(predictSet[k] == valSetT[k]):
                    accCount += 1
        if(accCount / len(valSet) * 100 > currMax):
            currMax = accCount / len(valSet) * 100
            maxC = i
            maxG = j            

    #use argmax to predict data and add the accuracy to the total    
    predictSet = []
    accCount = 0
    rbf = svm.SVC(class_weight = w, kernel='rbf', C = c[maxC], gamma = g[maxG])
    rbf.fit(trainingSet,trainingSetT)    
    predictSet = rbf.predict(testSet)    
   
    for i in range(len(predictSet)):
        if(predictSet[i] == testSetT[i]):
            accCount += 1
            
    accTotalRBFW += accCount / len(predictSet) * 100
    

    #Linear
    
    #Hyperparamater validation for c and return argMax
    currMax = 0
    for i in range(len(c)):
        predictSet = []
        accCount = 0
        linear = svm.SVC(kernel ='linear', C = c[i])
        linear.fit(trainingSet,trainingSetT)
        predictSet = linear.predict(valSet)
        
        for j in range(len(predictSet)):
            if(predictSet[k] == valSetT[k]):
                accCount += 1
        if(accCount / len(valSet) * 100 > currMax):
            currMax = accCount / len(valSet) * 100
            maxC = i
    
    #use argmax to predict data and add the accuracy to the total
    predictSet = []
    accCount = 0
    linear = svm.SVC(class_weight = w, kernel='linear', C = c[maxC])
    linear.fit(trainingSet,trainingSetT)
    predictSet = linear.predict(testSet)    
   
    for i in range(len(predictSet)):
        if(predictSet[i] == testSetT[i]):
            accCount += 1
            
    accTotalLinW += accCount / len(predictSet) * 100
    
  
    #Polynomial
    
    #Hyperparamater validation for c, gamma and degree, and return argMax
    currMax = 0
    for i in range(len(c)):
        for j in range(len(g)):
            for k in range(len(d)):            
                predictSet = []
                accCount = 0
                polynomial = svm.SVC(kernel='poly', C = c[i], gamma = g[j], degree = d[k])
                polynomial.fit(trainingSet,trainingSetT)
                predictSet = polynomial.predict(valSet)
        
                for l in range(len(predictSet)):
                    if(predictSet[l] == valSetT[l]):
                        accCount += 1
                if(accCount / len(valSet) * 100 > currMax):
                    currMax = accCount / len(valSet) * 100                    
                    maxC = i
                    maxG = j
                    maxD = k
            
    #use argmax to predict data and add the accuracy to the total        
    predictSet = []
    accCount = 0
    polynomial = svm.SVC(class_weight = w, kernel='poly', C = c[maxC], gamma = g[maxG], degree = d[maxD])
    polynomial.fit(trainingSet,trainingSetT)
    predictSet = polynomial.predict(testSet)    
   
    for i in range(len(predictSet)):
        if(predictSet[i] == testSetT[i]):
            accCount += 1
            
    accTotalPolyW += accCount / len(predictSet) * 100  
    
    
    #Sigmoid
    
    #Hyperparamater validation for c, gamma and coef, and return argMax
    currMax = 0
    for i in range(len(c)):
        for j in range(len(g)):
            for k in range(len(coef)):
                predictSet = []
                accCount = 0
                sigmoid = svm.SVC(kernel='sigmoid', C = c[i], gamma = g[j], coef0 = coef[k])
                sigmoid.fit(trainingSet,trainingSetT)
                predictSet = sigmoid.predict(valSet)
        
                for l in range(len(predictSet)):
                    if(predictSet[l] == valSetT[l]):
                        accCount += 1
                if(accCount / len(valSet) * 100 > currMax):
                    currMax = accCount / len(valSet) * 100
                    maxC = i
                    maxG = j       
                    maxCoef = k                       
    
    #use argmax to predict data and add the accuracy to the total
    predictSet = []
    accCount = 0
    sigmoid = svm.SVC(class_weight = w, kernel='sigmoid', C = c[maxC], gamma = g[maxG], coef0 = coef[maxCoef])
    sigmoid.fit(trainingSet,trainingSetT)
    predictSet = sigmoid.predict(testSet)    
   
    for i in range(len(predictSet)):
        if(predictSet[i] == testSetT[i]):
            accCount += 1
            
    accTotalSigW += accCount / len(predictSet) * 100 
    
    
#print the ouput data for all SVMs
print('One V.S. One')
print()
print('RBF prediction accuracy is: {}'.format(accTotalRBF/5))
print('Linear prediction accuracy is: {}'.format(accTotalLin/5))
print('Polynomial prediction accuracy is: {}'.format(accTotalPoly/5))
print('Sigmoid prediction accuracy is: {}'.format(accTotalSig/5))
print()
print()
    
print('One V.S. All')
print()
print('OvA prediction accuracy is: {}'.format(accTotalOVA/5))
print()
print()

print('Weighted')
print()
print('RBF prediction accuracy is: {}'.format(accTotalRBFW/5))
print('Linear prediction accuracy is: {}'.format(accTotalLinW/5))
print('Polynomial prediction accuracy is: {}'.format(accTotalPolyW/5))
print('Sigmoid prediction accuracy is: {}'.format(accTotalSigW/5))
print()
print()

#setup the plot of data
#setup the values
left = [1,2,3,4,5,6,7,8,9]
height = [accTotalRBF/5, accTotalLin/5, accTotalPoly/5, accTotalSig/5, 
accTotalOVA/5, accTotalRBFW/5, accTotalLinW/5, accTotalPolyW/5, accTotalSigW/5]
#setup the label names and colors
tick_label = ['RBF', 'Lin', 'Poly', 'Sig', 
'OvA', 'RBFW', 'LinW', 'PolyW', 'SigW']
plt.bar(left, height, tick_label = tick_label, width = 0.8, color = ['red', 'red', 'red', 'red', 'green', 'blue', 'blue', 'blue', 'blue'])
#setup the axis names and title, then show the bar graph
plt.xlabel('SVM types')
plt.ylabel('Accuracy (%)')
plt.title('SVM')
plt.show() 