import torch
from transformers import AutoImageProcessor
import datasets

dataset = datasets.load_dataset("sasha/dog-food")
# from PIL import Image
# image = Image.open(requests.get(url, stream=True).raw)

CHECKPOINT = "facebook/vit-mae-base"
processor = AutoImageProcessor.from_pretrained(CHECKPOINT)


def getBatch(idx: int, batch_size: int, minibatch_size: int):
    images = processor(images=dataset["train"][idx*batch_size*minibatch_size:(idx+1)*batch_size*minibatch_size]["image"], return_tensors="pt")["pixel_values"]
    batch = torch.unflatten(images, dim=0, sizes=(batch_size, minibatch_size))
    return batch


from svit.modelling import ViTForSimilarity

model = ViTForSimilarity.from_pretrained(CHECKPOINT)
print(model.config)
print(model(
    getBatch(0, 2, 5),
    getBatch(1, 2, 5),
    torch.Tensor([[1]*5]*2)
))