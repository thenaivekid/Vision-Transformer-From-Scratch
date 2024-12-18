# Vision Transformer (ViT) Project

The Vision Transformer (ViT) has become a foundational architecture for both image and video processing tasks. Inspired by the seminal paper *"An Image is Worth 16x16 Words"* ([arXiv:2010.11929](http://arxiv.org/abs/2010.11929)), this project aims to recreate the ViT model from scratch, train it, and extend it with additional experiments and enhancements.

## Project Objectives

### Completed
- ✅ **Load MNIST Dataset**  
  Successfully loaded the MNIST dataset for training and evaluation.  
- ✅ **Implement the Paper in Code**  
  Recreated the architecture described in the paper.  
- ✅ **Train the Model from Scratch**  
  Trained the model using supervised learning for a few epochs.  
- ✅ **Visualize Loss and Gradients**  
  Monitored loss curves and gradient flow to verify convergence during training.  

### In Progress
- ⬜ **Visualize Learned Positional Embeddings**  
  Examine the positional embeddings learned during training to understand their role in feature representation.  
- ⬜ **Visualize Attention Maps**  
  Generate and analyze attention maps for selected images to interpret how the model attends to different regions.  
- ⬜ **Load Pretrained Weights**  
  Incorporate pretrained weights into the model for transfer learning.  
- ⬜ **Finetune for 2-Class Classification**  
  Adapt the model to classify two specific classes plus an additional "unknown" category.  
- ⬜ **Visualize Clustering Effect**  
  Analyze the clustering of features in the latent space for the two classes.  

### Advanced Objectives
- ⬜ **Self-Supervised Training**  
  Explore self-supervised training methods, such as Masked Autoencoding (MAE) and Jigsaw Epistemic Perturbation Aggregation (JEPA) (optional).  
- ⬜ **Distill the Model**  
  Reduce the model size through knowledge distillation while maintaining performance.  
- ⬜ **Enhance with Synthetic Data**  
  Use synthetic datasets to improve model generalization and robustness.  

## Architecture Overview

The ViT architecture partitions an input image into patches, treats them as tokens, and processes them through a transformer encoder.  
![Model Architecture](https://raw.githubusercontent.com/SHI-Labs/Compact-Transformers/main/images/model_sym.png)

## Training on MNIST

The model was trained on the MNIST dataset, with results visualized during training:  
![Training Visualization](https://github.com/thenaivekid/Vision-Transformer-From-Scratch/blob/main/viz.png)

## References
- [An Image is Worth 16x16 Words (arXiv:2010.11929)](http://arxiv.org/abs/2010.11929)  
- [MNIST Dataset (Hugging Face)](https://huggingface.co/datasets/ylecun/mnist)  
