import numpy as np

def normalizeData(data: np.array) -> np.array:
	from sklearn.preprocessing import StandardScaler
	
	data_scaler = StandardScaler()
	return data_scaler.fit_transform(data)

data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
print(data)
normalized_data = normalizeData(data)
print(normalized_data)