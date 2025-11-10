import torch
print("cuda.is_available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available(): print("name:", torch.cuda.get_device_name(0))
print("version:", torch.version.cuda)
