from NBAPlayer import *
import sys
import math
import copy  
import csv

sampleData = []  
sampleDataSize = int(sys.argv[4])
testData = []
positions = ['C', 'PF', 'SF', 'SG', 'PG']

mean = NBAPlayer()
std_deviation = NBAPlayer()

attributes = open(sys.argv[3], 'r').read().strip().split("\n")

def myKnn(sampleData, testData, k):
    accuracy = 0
    if k > len(sampleData):
        k = len(sampleData)
    p_info = []
    for testPlayer in testData:
        small = 99999.99
        p_info = []
        for player in sampleData:
            p_dist = 0
            for attribute in attributes:
                p_dist = p_dist + ((testPlayer.data["normal_"+attribute] - player.data["normal_"+attribute]) ** 2)
            if p_dist < small:
                player.data["score"] = p_dist
                p_info.append(player)

                max1_pos = 0
                max1 = p_info[0].data["score"]
                max2 = 0
                for i in range(1,len(p_info)):
                    if max1 < p_info[i].data["score"]:
                        max2 = max1
                        max1 = p_info[i].data["score"]
                        max1_pos = i
                    elif max2 < p_info[i].data["score"]:
                        max2 = p_info[i].data["score"]
                if len(p_info) > k:
                    small = max2
                    p_info.pop(max1_pos)
                elif len(p_info) == k:
                    small = max1
                
        C_counter = 0
        PF_counter = 0
        SG_counter = 0
        SF_counter = 0
        PG_counter = 0
        max_counter = 0
        max = ''
        for i in range(0, len(p_info)):
            if p_info[i].data["Pos"].upper() == "C":
                C_counter = C_counter + 1
                if C_counter > max_counter:
                    max_counter = C_counter
                    max = "C"
            elif p_info[i].data["Pos"].upper() == "PF":
                PF_counter = PF_counter + 1
                if PF_counter > max_counter:
                    max_counter = PF_counter
                    max = "PF"
            elif p_info[i].data["Pos"].upper() == "PG":
                PG_counter = PG_counter + 1
                if PG_counter > max_counter:
                    max_counter = PG_counter
                    max = "PG"
            elif p_info[i].data["Pos"].upper() == "SF":
                SF_counter = SF_counter + 1
                if SF_counter > max_counter:
                    max_counter = SF_counter
                    max = "SF"
            elif p_info[i].data["Pos"].upper() == "SG":
                SG_counter = SG_counter + 1
                if SG_counter > max_counter:
                    max_counter = SG_counter
                    max = "SG"
        if testPlayer.data["Pos"].strip().upper() == max:
            accuracy = accuracy + 1
    accuracy = float(accuracy)/len(testData)
    return accuracy

def iterateKmeans(sampleData, clusters, change):
    if change == 0:
        return clusters
    else:
        change = 0
        for cluster in clusters:
            cluster.clusterData = []

        for player in sampleData:
            small = 99999.99
            c_info = None
            for cluster in clusters:
                c_dist = 0
                for attribute in attributes:
                    c_dist = c_dist + ((cluster.data["normal_"+attribute] - player.data["normal_"+attribute]) ** 2)
                if small >= c_dist:
                    small = c_dist
                    c_info = cluster
            c_info.clusterData.append(player)
        
        for cluster in clusters:
            for attribute in attributes:
                sum_attr_value = 0
                for cluster_player in cluster.clusterData:
                    sum_attr_value = sum_attr_value + cluster_player.data["normal_"+attribute]
                if len(cluster.clusterData) > 0 and round((sum_attr_value / len(cluster.clusterData)),2) != cluster.data["normal_"+attribute]:
                        change = 1
                        cluster.data["normal_"+attribute] = round(sum_attr_value/len(cluster.clusterData), 2)
        
        iterateKmeans(sampleData, clusters, change)

def myKmeans(sampleData, k):
    
    clusters = []
    
    for i in range(0,k):
        cluster = NBAPlayer()
        cluster.assignRandom(i, attributes)
        cluster.clusterData = []
        clusters.append(cluster)

    iterateKmeans(sampleData, clusters, 1)
    
    for cluster in clusters:

        cluster.posData = []
        cluster.data["SG"] = 0
        cluster.data["SF"] = 0
        cluster.data["PG"] = 0
        cluster.data["PF"] = 0
        cluster.data["C"] = 0
        
        for player in cluster.clusterData:
            if player.data["Pos"].upper() == "C":
                cluster.data["C"] = cluster.data["C"] + 1
            elif player.data["Pos"].upper() == "PF":
                cluster.data["PF"] = cluster.data["PF"] + 1
            elif player.data["Pos"].upper() == "PG":
                cluster.data["PG"] = cluster.data["PG"] + 1
            elif player.data["Pos"].upper() == "SF":
                cluster.data["SF"] = cluster.data["SF"] + 1
            elif player.data["Pos"].upper() == "SG":
                cluster.data["SG"] = cluster.data["SG"] + 1
    return clusters
    
# To normalize the data points
def Normalize():
    for player in sampleData:
        for attribute in attributes:
            player.data["normal_"+attribute] = round((player.data[attribute] - mean.data[attribute])/std_deviation.data[attribute], 2)
    
    for player in testData:
        for attribute in attributes:
            player.data["normal_"+attribute] = round((player.data[attribute] - mean.data[attribute])/std_deviation.data[attribute], 2)

    if sys.argv[1] == "kmeans":
        clusters = myKmeans(sampleData, int(sys.argv[2]))
        for cluster in clusters:
            print
            print cluster.data["PlayerName"]
            print "-- No. of Players: " + str(len(cluster.clusterData))
            print "-- Cluster Center: "
            for attribute in attributes:
                print "---- " + attribute + " : " + str(cluster.data["normal_"+attribute])
            print "-- Distribution of positions: "
            for position in positions:
                print "---- " + position + " : " + str(cluster.data[position])
            
    else:
        c = myKnn(sampleData, testData, int(sys.argv[2]))
        print "Accuracy: " + str(c)

# To find standard deviation for respective attributes 
def deriveStatVariables(sum_attr):
    sampleDataSize = len(sampleData)

    # Storing means of all variables
    for attribute in attributes:
        mean.data[attribute] = sum_attr.data[attribute]/sampleDataSize
    del sum_attr
    
    sum_attr = NBAPlayer()
    for player in sampleData:
        for attribute in attributes:
            if(sum_attr.data[attribute] == -1):
                sum_attr.data[attribute] = 0
            sum_attr.data[attribute] = sum_attr.data[attribute] + ((player.data[attribute] - mean.data[attribute]) ** 2)

    for attribute in attributes:
        std_deviation.data[attribute] = (sum_attr.data[attribute] / sampleDataSize) ** 0.5
    
    Normalize()

# Reading of CSV Data and adding them to sample and test data
with open('NBAStats.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')

    # Counter for dividing data into sample and test
    i = 0

    # Preparing Sample Data
    for row in readCSV:
        player = None
        if i != 0:
            player = NBAPlayer()
            if player.assignData(row) == True:
                if i > sampleDataSize:
                    testData.append(player)
                else:
                    sampleData.append(player)
                    
                    for attribute in attributes:
                        sum_attr.data[attribute] = sum_attr.data[attribute] + player.data[attribute]                
            else:
                exit('Issue with respect to Number of columns. Check row: '+row)
                
        elif i == 0:
            sum_attr = NBAPlayer()
            for attribute in attributes:
                sum_attr.data[attribute] = 0
        i = i + 1
    
    deriveStatVariables(sum_attr)