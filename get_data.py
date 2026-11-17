import numpy as np

from retrieval.library import AEMLibraryWriter

model_matrix = np.load("./pre.npy")
height_values = np.load("./h.npy")
data = np.load("./data.npy")
output = './tests/filed'
writer = AEMLibraryWriter(
    output,
    capacity=model_matrix.shape[0],
    n_model_parameters=model_matrix.shape[1],
)

for index in range(writer.count, model_matrix.shape[0]):
    height = height_values[index]
    response = data[index]
    model = model_matrix[index]
    writer.add(
        response,
        height_m=height,
        model=model
    )
writer.finalize()
writer.release()
