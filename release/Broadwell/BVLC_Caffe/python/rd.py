import pprint, pickle
import numpy as np 

pkl_file = open('conv1_diff_after_bp_0.pkl', 'rb')
#pkl_file = open('class_map_0.2.pkl', 'rb')
data1 = pickle.load(pkl_file)
#pprint.pprint(data1)
#print data1
pkl_file.close()

pkl_file = open('conv1_diff_after_bp.pkl', 'rb')
#pkl_file = open('class_map_0.2.pkl', 'rb')
data = pickle.load(pkl_file)
#pprint.pprint(data1)
#print data
pkl_file.close()
print data.shape
print np.array_equal(data[0][0], data1[0][0])
print data[0][0]
print data1[0][0]
