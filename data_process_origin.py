import numpy as np 
import glob
import os 
from scipy import spatial 
import pickle
import pandas as pd

# Please change this to your location
data_root = './data/'
frame_id = 0
object_id = 1
object_type = 2
position_x = 3
position_y = 4
position_z = 5
object_length = 6
object_width = 7
object_height = 8
heading = 9


def get_origin_data_list(pra_file_path):
	'''
	Read raw data from files and return a dictionary:
		{frame_id:
			{object_id:
				# 10 features
				[frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading]
			}
		}
	'''
	# print(train_file_path)
	frame_list = []
	content = pd.read_csv(pra_file_path, header=None).to_numpy()
	content = content[:, [1, 0, 4, 20, 21, 22, 52, 54, 56, 50]]
	map_id = {}
	id_count = 1
	for row in content:
		if not (row[1] in map_id):
			map_id[row[1]] = id_count
			id_count += 1
		row[1] = map_id[row[1]]
		frame_list.append(row)
	return frame_list


def generate_origin_data(file_path_list):
	'''
	Read data from $pra_file_path, and split data into clips with $total_frames length.
	Return: feature and adjacency_matrix
		feture: (N, C, T, V)
			N is the number of training data
			C is the dimension of features, 10raw_feature + 1mark(valid data or not)
			T is the temporal length of the data. history_frames + future_frames
			V is the maximum number of objects. zero-padding for less objects.
	'''

	result = np.array([])
	for file_path in file_path_list:
		now_list = get_origin_data_list(file_path)
		result = np.array(now_list)

	pd.DataFrame(result).to_csv('sample.csv', sep=' ', index=False, header=False)


if __name__ == '__main__':
	origin_data_file_path_list = sorted(glob.glob(os.path.join(data_root, '*.csv')))
	print('Generating Origin Data.')
	generate_origin_data(origin_data_file_path_list)



