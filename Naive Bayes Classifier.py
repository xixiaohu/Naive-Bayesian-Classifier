import csv
import sqlite3
import matplotlib.pyplot as plt

#connect to SQL database
conn = sqlite3.connect( 'flyingfit.db' ) 
c = conn.cursor()

#if I want to run several times,  the table I built before should be dropped first
c.execute( 'DROP TABLE flyingfitness' ) 
c.execute('CREATE TABLE flyingfitness (obs INTEGER, testRes INTEGER, var2 INTEGER, var3 INTEGER, var4 INTEGER, var5 INTEGER, var6 INTEGER)') 

#read csv data, insert into SQL database
with open('Flying_Fitness.csv', newline='\n') as ff:
	ds = csv.reader(ff, delimiter=',', quotechar='"')
	for row in ds:
		c.execute('INSERT INTO flyingfitness VALUES(?, ?, ?, ?, ?, ?, ?)', tuple(row))

#select data from the table and store it into a dataset 
c.execute ('SELECT * from flyingfitness')
rawdata = c.fetchall() 
conn.commit()
conn.close()

#convert tuples into lists
dataset = [list(row) for row in rawdata] 
#delete headers
del dataset[0] 
#convert elements into floating numbers
for i in range(len(dataset)):
	dataset[i] = [int(f) for f in dataset[i]]
	#delete first column (obs #)
	del dataset[i][0] 

#separate data into dictionary 
def separatedata(dataset,index):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i] 
		#separate data by index
		if (vector[index] not in separated):
			#create a list for new value
			separated[vector[index]] = []
		#append each line for its values
		separated[vector[index]].append(vector) 
	return separated

#separate dataset into classes
sepdatac = separatedata(dataset,0)

#calculate probabilities of given dictrionary
def prob(dictionary):
	count = {} #count number of records in each class / variable
	for key, values in dictionary.items():
		count[key] = len(values)
	totrecords = sum(count.values()) #number of total records
	clprob = {} #calculate probablies 
	for key, values in count.items():
		clprob[key] = values/totrecords
	return clprob 

clsprob = prob(sepdatac) #class probability

#calculate probabilities of given class
def varcondprob(dictionary,index):
	classdata = dictionary.get(index) #create a dataset for given class
	#calculate conditional probablities for each variable of given class
	condprob = {}
	for variables in range(1,len(classdata[0])):
		if (variables not in condprob):
			condprob[variables] = {}
		vardata = separatedata(classdata,variables) 
		varcondprob = prob(vardata)
		condprob[variables] = varcondprob	
	return condprob

#conditional probabilities for all cases
def condprob(classdict): 
	condprob = {}
	for key in classdict:
		if (key not in condprob):
			condprob[key] = {}
		prob = varcondprob(classdict, key)
		condprob[key] = prob
	return condprob

#conditional probabilities for all cases
condprob = condprob(sepdatac)

#predict class by choose max probability and calculate prediction score (probability of 1)
def predclass(dataset, classprob, condprob):
	prediction = []
	predscore = []
	for l in dataset:
		prediction.append(0) #create a list of 0 with length = lines in the dataset
		
	for line in range(0,len(dataset)): #each line
		p = 0
		predsum = 0
		predone = 0
		for key in classprob: #probabilities for given class
			prob = classprob[key] #P(class = 0/1)
			for var in range(1,len(dataset[0])): #each column
				point = dataset[line][var]
				if (point not in condprob[key][var]):
					prob = prob*0  #value of given variable cannot get given class, P(variable|clss)=0
				else:
					probpoint = condprob[key][var][point]
					prob = prob*probpoint #P(variable|class)
			if prob > p: #max probability
				p = prob
				prediction[line] = key #attach class value to the prediction list
			else:
				p = p
						
			predsum += prob #total probability
			if key == 1: #check if true class is 1
				predone = prob
		pscore = predone / predsum #calculate the prediction score for ROC curve
		predscore.append(pscore) #list of prediction score for p(1)
	return prediction, predscore

prediction = predclass(dataset, clsprob, condprob)[0]
predscore = predclass(dataset, clsprob, condprob)[1]

#get true class value
trueclass = []
for l in dataset:
	trueclass.append(l[0])

#sort class & predscore based on predscore
predscore, trueclass = (list(x) for x in zip(*sorted(zip(predscore, trueclass))))

#calculate tpr & fpr for given list
def rate(tclass, total1, total0):
	count0 = 0
	count1 = 0
	tpr = 0
	fpr = 0
	for item in tclass:
		if item == 0:
			count0 += 1
		elif item == 1:
			count1 += 1
	tpr = count1 / total1
	fpr = count0 / total0
	return tpr, fpr


total1 = len(sepdatac[1]) #get total number of true class = 1 in dataset
total0 = len(sepdatac[0]) #get total number of true class = 0 in dataset

#get the points of ROC curve by using prediction score as threshold
tprl = []
fprl = []
i = 0
while i in range(0, len(predscore)): 
	tclass = []
	if predscore[i] == 0: #If the threshold equals to 0, tpr & fpr should be (1,1)
		tprl.append(1)
		fprl.append(1)
		i += 1
	elif predscore[i] == 1: #If the threshold equals to 1, tpr & fpr should be (0,0)
		tprl.append(0)
		fprl.append(0)
		i += 1
	elif predscore[i] == predscore[i+1]: #If the threshold equals to next one, move to next line
		i += 1
	else:
		for p in range(i+1, len(trueclass)): #get the list of the rest records to calculate rates
			tclass.append(trueclass[p])
		rates = rate(tclass, total1, total0) #calculate rate
		tpr = rates[0]
		tprl.append(tpr) #get a list of tpr
		fpr = rates[1]
		fprl.append(fpr) #get a list of fpr
		i += 1

plt.fill_between(fprl, tprl, color='lightgrey', label='AUC') #draw the AUC area
plt.plot(fprl, tprl, color='darkturquoise', linewidth=2, label='ROC Curve') #draw the ROC curve
plt.scatter(fprl, tprl, color='orangered', marker='*')
plt.plot([0,1], [0,1], color='peachpuff', ls = '--', linewidth=2, label='Random') #draw the random cuve 
plt.axis([0, 1, 0, 1])
plt.legend()
plt.title('ROC Curve & AUC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()











