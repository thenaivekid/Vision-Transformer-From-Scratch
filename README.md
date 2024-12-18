# Vision Transformer

VITs is a widely used architecture in both image and video processing. That's why I wanted to recreate it myself and train from scratch. 
But this project does not stop there. Following are my objectives:
- [x] load dataset
- [x] implement paper in code.
- [x] train from scratch for a few epochs supervised(training)
- [x] viz loss and gradients to see if the training is approaching a minima or not
- [ ] viz learned positional embedding
- [ ] viz attn map for few images
- [ ] load weights from pretrained
- [ ] finetune for 2 class classification. (2 class plus 1 unknown)
- [ ] viz clustering effect of vit in 2 classes
- [ ] do everything by doing self supervised training. mae, jepa(optional)
- [ ] distill this in smaller model
- [ ] use synthetic data to make it better

## The architecture:
!(https://raw.githubusercontent.com/SHI-Labs/Compact-Transformers/main/images/model_sym.png)[]

