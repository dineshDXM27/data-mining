import numpy as np
import math
import sys
import matplotlib.pyplot as plt 
import time

# Function to generate data set of 2 classes based on given data
def generateDataset(size):

    # Defining the multivariant means, variance and class
    class1 = {
        "mean": [1, 0],
        "variance": [[1, 0.75],[0.75, 1]],
        "class": 0
    }

    class2 = {
        "mean": [0, 1.5],
        "variance": [[1, 0.75],[0.75, 1]],
        "class": 1
    }

    # Generating dataset Data
    a = np.random.multivariate_normal(class1["mean"], class1["variance"], int(size/2))
    b = np.random.multivariate_normal(class2["mean"], class2["variance"], int(size/2))

    # Adding classes
    a = np.append(a, np.full((int(size/2), 1), class1["class"]), 1)    
    b = np.append(b, np.full((int(size/2), 1), class2["class"]), 1)

    dataset = np.concatenate((a, b))        # Concatination both the arrays
    dataset = np.append(np.full((size,1), 1), dataset, 1)     # Adding Bias Term
    np.random.shuffle(dataset)      # Shuffling 
    
    return dataset

# Function to return sigmod for given weights and attribute values
def sigmoid(w, x):
    z = np.sum(w*x[:3])
    try: 
        sig = (1/(1+math.exp(-z)))
        return sig
    except:
        return 0

# Main Function    
def __main__(argv):
    
    allFlag = False     # Flag to perform all methods along with learning rates of 0.01, 0.1 and 1
    batchFlag = False   # Flag to know if the user is looking for BATCH processing
    
    

    if len(argv) > 1 and 'roc' != argv[1] and 'ROC' != argv[1]: 

        if argv[3]:
            try:
                size = int(argv[3])
            except:
                exit('Size should be defined as integer')

        if argv[1].lower() == "batch":
            batchFlag = True
        elif argv[1].lower() == "online":
            batchFlag = False
        else:
            exit('Follow the syntax \n python main.py [BATCH / ONLINE] [LEARNING_RATE]')

        if argv[2]:
            try:
                learning_rate = float(argv[2])
            except: 
                exit('LEARNING RATE should be defined as numbers')
        else:
            exit('Follow the syntax \n python main.py [BATCH / ONLINE] [LEARNING_RATE]')
    else:
        size = 2000             # Static size 
        allFlag = True

    # Generating Training and Testing Dataset
    trainingDataSet = generateDataset(size)
    testingDataSet = generateDataset(size/2)

    # Setting Threshold for the gradients to converge
    threshold = 0.0000001

    if allFlag == True:
        for l in [0.01, 0.1, 1]:
            runLogisticRegression(True, threshold, l, trainingDataSet, testingDataSet)
            runLogisticRegression(False, threshold, l, trainingDataSet, testingDataSet)
        
    else:
        runLogisticRegression(batchFlag, threshold, learning_rate, trainingDataSet, testingDataSet)

# Function to run a complete cycle of Logistic Regression based on given parameters
def runLogisticRegression(batchFlag, threshold, learning_rate, trainingDataSet, testingDataSet):
    print "-----------------------------------------"
    if batchFlag:
        print "Type: Batch Processing with learning rate: " + str(learning_rate)
        weight = np.full((3), 1, dtype=float)
        start = time.time()
        weight = batch(threshold, learning_rate, weight, trainingDataSet)

    else:
        print "Type: Online Processing with learning rate: " + str(learning_rate)
        weight = np.full((3), 1, dtype=float)
        start = time.time()
        weight = online(threshold, learning_rate, weight, trainingDataSet)
    print "Time(s): " + str(round(time.time() - start, 3))
    print "Weights: " + str(weight)
    testData(weight, testingDataSet, batchFlag, learning_rate)
    print

# Function to run training based on online processing in Logistic Regression
def online(threshold, learning_rate, weight, trainingDataSet):
    interationCounter = 0
    while(True):
        if interationCounter == 10000:
            print "Iteration(s): " + str(interationCounter)
            return weight
        interationCounter += 1
        pError = 0
        for x in trainingDataSet:
            o = sigmoid(weight, x)
            gradient = (-learning_rate * (o - x[3]) * x[:3])
            weight = weight + gradient

            # Cross Entropy
            try: 
                Error = - x[3]*math.log(o) 
            except:
                Error = 0
            try:
                Error = Error - (1-x[3])*math.log(1-o)
            except:
                Error = Error

            # Checking of Cross Entropies
            if round(Error,6) == round(pError,6):
                print "Iteration(s): " + str(interationCounter)
                return weight
            pError = Error

            # Checking for convergence of gradients
            if abs(gradient[0]) <= threshold and abs(gradient[1]) <= threshold and abs(gradient[2]) <= threshold:
                print "Iteration(s): " + str(interationCounter)
                return weight

# Function to run training based on batch processing in Logistic Regression
def batch(threshold, learning_rate, weight, trainingDataSet):
    size = len(trainingDataSet)
    interationCounter = 0
    pError = -1
    while(True):
        if interationCounter == 10000:
            print "Iteration(s): " + str(interationCounter)
            return weight
        interationCounter += 1
        sum_weights = np.zeros((3), dtype=float)

        Sigmoid = sigmoidForNumpyArray(trainingDataSet, weight)     # Sigmoid of complete NUMPY array
        sum_weights = (np.dot((Sigmoid - trainingDataSet[:,3]),trainingDataSet[:,:3]))
        gradient = (-learning_rate * (sum_weights/size))
        weight = weight + gradient

        # Cross Entropy
        Error = 0
        for i in range(0, len(trainingDataSet)):

            # Cross Entropy
            try: 
                Error = Error - trainingDataSet[i][3]*math.log(Sigmoid[i]) 
            except:
                Error = Error
            try:
                Error = Error - (1-trainingDataSet[i][3])*math.log(1-Sigmoid[i])
            except:
                Error = Error
        
        # Checking of Cross Entropies
        if round(Error,6) == round(pError,6):
            print "Iteration(s): " + str(interationCounter)
            return weight
        pError = Error

        # Checking for convergence of gradients
        if abs(gradient[0]) <= threshold and abs(gradient[1]) <= threshold and abs(gradient[2]) <= threshold:
            print "Iteration(s): " + str(interationCounter)
            print "Error: " + str(Error)
            return weight

# Generic Function to test Weights generated with the testing Dataset and plotting the ROC Graphs  
def testData(weight, dataSet, batchFlag, learning_rate):
    final = sortBasedOnActivation(dataSet, sigmoidForNumpyArray(dataSet, weight))
    accuracy = 100*(float(np.count_nonzero(np.equal(final[:,3], final[:,4])))/len(dataSet))
    print "Accuracy = " + str(accuracy)
    if 'roc' in sys.argv or 'ROC' in sys.argv:
        plotGraph(final, batchFlag, learning_rate)

# Returns Sigmoid of the dataSet given along with weight
def sigmoidForNumpyArray(dataSet, weight):
    Z = (np.sum(dataSet[: , :3]*weight, axis = 1, dtype = float))   # Finding sum 'Z'
    Sigmoid = 1/(1+np.exp(-1*Z))
    return Sigmoid

# Adding Columns and sorting based on activation
# Structure [ bias_term, x1_term, x2_term, Actual, Sigmoid_Result, Sigmoid_Activation_Value ]
def sortBasedOnActivation(dataSet, Sigmoid):
    dataSet = np.column_stack((dataSet, np.rint(Sigmoid), Sigmoid))
    return dataSet[dataSet[:,5].argsort()]

# Function to help generate ROC Curve and find Area under the curve
def plotGraph(dataSet, batchFlag, learning_rate):
    actual_pos = np.count_nonzero(dataSet[:,3]==1)

    actual_neg = np.count_nonzero(dataSet[:,3]==0)
    points = np.array([[1,1]])

    for activationThreshold in np.concatenate(([0],dataSet[:,5])):
        temp = dataSet[np.where(dataSet[:,5] >= activationThreshold)]
        match = np.equal(temp[:,3],temp[:,4])
        true = temp[match]
        tpr = float(len(true[np.equal(true[:,3], 1)]))/actual_pos

        false = temp[np.logical_not(match)]
        fpr = float(len(false[np.equal(false[:,3], 0)]))/actual_neg
        points = np.append(points, [[tpr, fpr]], axis=0)

    plt.plot(points[:,1], points[:,0])
    A = 1 + np.trapz(points[:,1],x=points[:,0])

    if batchFlag:
        method = "BATCH"
    else:
        method = "ONLINE"
    plt.title("ROC Curve for " + method + " under learning rate of " + str(learning_rate))
    plt.xlabel("FPR" + "\n Area under the curve: " + str(A))
    plt.ylabel("TPR")
    print "Area under the curve: " + str(A)
    plt.show()

__main__(sys.argv)