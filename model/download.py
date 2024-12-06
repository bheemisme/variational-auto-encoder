import kagglehub
import os

path = kagglehub.model_download("sudarshan1927/variational-auto-encoder-fashion-minist/pyTorch/default")
model_path = os.path.join(path, "vae_model_train_27_epoch_100.pth")
print(model_path)
