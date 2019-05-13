import random

class NBAPlayer:
    def __init__(self):
        self.data = {
            "PlayerName": None,
            "Pos": None,
            "Age": -1,
            "Tm": None,
            "G": -1,
            "GS": -1,
            "MP": -1,
            "FG": -1,
            "FGA": -1,
            "FG_percent": -1,
            "_3P": -1,
            "_3PA": -1,
            "_3P_percent": -1,
            "_2P": -1,
            "_2PA": -1,
            "_2P_percent": -1,
            "eFG_percent": -1,
            "FT": -1,
            "FTA": -1,
            "FT_percent": -1,
            "ORB": -1,
            "DRB": -1, 
            "TRB": -1,
            "AST": -1,
            "STL": -1,
            "BLK": -1,
            "TOV": -1,
            "PF": -1,
            "PS_by_6": -1
        }

    def assignData(self, record):
        if len(record) == 29:
            self.data["PlayerName"] = record[0]
            self.data["Pos"] = record[1]
            self.data["Age"] = float(record[2])
            self.data["Tm"] = record[3]
            self.data["G"] = float(record[4])
            self.data["GS"] = float(record[5])
            self.data["MP"] = float(record[6])
            self.data["FG"] = float(record[7])
            self.data["FGA"] = float(record[8])
            self.data["FG_percent"] = float(record[9])
            self.data["_3P"] = float(record[10])
            self.data["_3PA"] = float(record[11])
            self.data["_3P_percent"] = float(record[12])
            self.data["_2P"] = float(record[13])
            self.data["_2PA"] = float(record[14])
            self.data["_2P_percent"] = float(record[15])
            self.data["eFG_percent"] = float(record[16])
            self.data["FT"] = float(record[17])
            self.data["FTA"] = float(record[18])
            self.data["FT_percent"] = float(record[19])
            self.data["ORB"] = float(record[20])
            self.data["DRB"] = float(record[21])
            self.data["TRB"] = float(record[22])
            self.data["AST"] = float(record[23])
            self.data["STL"] = float(record[24])
            self.data["BLK"]= float(record[25])
            self.data["TOV"] = float(record[26])
            self.data["PF"] = float(record[27])
            self.data["PS_by_6"] = float(record[28])
            return True
        else:
            return False

    def __getattribute__(self, attr):
        return self.data[attr]

    def assignRandom(self, i, attributes):
        self.data["PlayerName"] = "Cluster " + str(i+1)
        for attribute in attributes:
            self.data["normal_"+attribute] = round(float(random.random()),2)