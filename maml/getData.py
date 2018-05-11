import pickle
import numpy as np

#import cPickle as pickle

dataset_PATH = "data/"

task_id = "C-tanh"
# task_id = "C-sin"
# task_id = "bounce-states"
if task_id == "C-tanh":
    num_shots = 10


filename = dataset_PATH + task_id + "_{0}-shot.p".format(num_shots)
tasks = pickle.load(open(filename, "rb"))

#print(tasks)

myT = tasks["tasks_train"]

len(myT[0:25])
#We will split the 100 up into 4 batches of 25 each.
first_batch = myT#[0:25]

complete_in_a = np.array([[]])
complete_la_a = np.array([[]])
complete_in_b = np.array([[]])
complete_la_b = np.array([[]])

#Best way to do this is just do this for all. 

for i in xrange(0,100):
	#This is the count for which example.
	#i = 0 
	print("I:",i)
	#The first zero removes the dictionary of the true values.
	inputa = first_batch[i][0][0][0].ravel()
	print("inputa: " , inputa)
	labela = first_batch[i][0][0][1].ravel()

	inputb = first_batch[i][0][1][0].ravel()
	labelb = first_batch[i][0][1][1].ravel()

	if i == 0:
		complete_in_a = inputa
		complete_la_a = labela
		complete_in_b = inputb
		complete_la_b = labelb
	else:
		complete_in_a = np.vstack((complete_in_a,inputa))
		complete_la_a = np.vstack((complete_la_a,labela))
		complete_in_b = np.vstack((complete_in_b,inputb))
		complete_la_b = np.vstack((complete_la_b,labelb))
		

complete_in_a = complete_in_a.reshape(-1,10,1)
complete_in_b = complete_in_b.reshape(-1,10,1)
complete_la_a = complete_la_a.reshape(-1,10,1)
complete_la_b = complete_la_b.reshape(-1,10,1)

print("final complete:")
print(complete_in_a)
print(complete_in_a.shape)


print("inputa: " , inputa)
print("inputb: " , inputb)

print('ina',complete_in_a)
print('inb',complete_in_b)
print('laa',complete_la_a)
print('lab',complete_la_b)
print(complete_in_a.shape)





