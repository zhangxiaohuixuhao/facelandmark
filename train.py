import torch
import numpy as np

image = np.ones((112, 112, 3))
end = torch.from_numpy(image)
print('ok')