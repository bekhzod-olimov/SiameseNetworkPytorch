# SiameseNetworkPytorch

Classify people with DL-based image classification model [efficientnetb0](https://github.com/lukemelas/EfficientNet-PyTorch) [(paper)](https://arxiv.org/pdf/1905.11946.pdf) trained using [Siamese Network](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf), test the model performance on unseen images during training.

### Create virtual environment
```python
conda create -n <ENV_NAME> python=3.9
conda activate <ENV_NAME>
pip install -r requirements.txt
```

### Sample data

![Capture1](https://user-images.githubusercontent.com/50166164/209507049-bca12b74-d80c-48cc-a78e-644db4422564.PNG)
![Capture2](https://user-images.githubusercontent.com/50166164/209507053-6306ebfd-af8d-4798-ba50-caa6815d637d.PNG)
![Capture](https://user-images.githubusercontent.com/50166164/209507054-fb7da7f4-9d58-4e30-b3b7-76a4f7e8e7f7.PNG)

### Run training 
```python
python train.py --batch_size=64 --lr=3e-4 --model_name="efficientnet_b3a"
```

### Results

![Capture3](https://user-images.githubusercontent.com/50166164/209507090-4dfb6608-ef74-4439-9173-8798c57a034d.PNG)
![Capture4](https://user-images.githubusercontent.com/50166164/209507094-511eead9-d3f9-4c18-a8b3-d6f79e1a26b4.PNG)
![Capture5](https://user-images.githubusercontent.com/50166164/209507095-7b5fbaca-e666-426b-b0b8-dadadad459d7.PNG)
