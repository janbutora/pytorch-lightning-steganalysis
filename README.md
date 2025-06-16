# Pytorch Lightning steganalysis

To run the training script, you can adjust the arguments, and run the script:

```
./train_lit_model.py --backbone efficientnet-b0 --batch-size 32 --version test --gpus 0,1 --epochs 20
```

Note that the file data/splits.npy should follow your data logic.
