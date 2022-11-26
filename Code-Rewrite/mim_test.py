import utils
import torch

img_batch = torch.randn(16, 3, 32, 32)
model = utils.get_gdumb_resnet_impl()
# Convert to feature extractor
model.final = torch.nn.Identity() # type: ignore
dim_h = model.dim_out
features = model(img_batch)

phi = torch.nn.Linear(dim_h, 128)
classifier = torch.nn.Linear(dim_h, 10)

projected = phi(features)
print(projected.shape, projected.square().sum().sqrt())

projected = torch.nn.functional.normalize(projected)
print(projected.shape, projected.square().sum().sqrt())

for i in range(projected.shape[0]):
    print(projected[i].detach().square().sum().sqrt())