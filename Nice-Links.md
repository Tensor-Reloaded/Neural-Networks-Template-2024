## Data Augmentation 

* Torchvision (standard and a good starting point): [link](https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html#sphx-glr-auto-examples-transforms-plot-transforms-getting-started-py).
* Albumentations (more advanced, good for object detection and segmentation): [link](https://github.com/albumentations-team/albumentations).
* Kornia (Batched data augmentation on GPU): [link](https://github.com/kornia/kornia).

## Training with mixed precision

* torch.autocast, GradScaler: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html.

## Hyperparameter tuning

* Use weights and biases (wandb): https://docs.wandb.ai/tutorials/sweeps/.

## Use Learning Rate Schedulers

* Reduce Learning Rate On Plateau: [link](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html).
* Step Learning Rate: [link](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html).
* Cosine Annealing Learning Rate: [link](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html).

## Pretrained models

* HuggingFace models: https://huggingface.co/models.
* PyTorch Image Models (timm):
  * [Github repo](https://github.com/huggingface/pytorch-image-models)
  * [timm on HuggingFace](https://huggingface.co/models?library=timm&sort=trending)
