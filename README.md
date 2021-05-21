# submission-352
This is the source code for submission-654 of CIKM 2021: PeriodicMove: Shift-aware Human Mobility Recovery with Graph Neural Network

## Dependencies

- Python==3.6
- torch==1.5.0
- transformers==2.9.0
- easydict==1.9
- matplotlib==3.1.1



## Running

For the simplification, we only supply the foursquare dataset used in the paper. 

```python
v3_delta_0.005_window_size_6_mask_num_10.tar.gz
```

Firstly, the user should extract the file, e.g., pos.vocab.txt. Then they need to change the setting in the config.

```python
vocab_path 						# the path of vocab file
dist_path							# the path of distance file
train_file 						# the path of traning data
eval_path							# the path of validation data
test_path							# the path of testing data
save_dir							# the path for saving model and embedding matrix
```

After that, type the following command in the termination.

```bash
python main.py
```

When the training procedure is completed, the termination will print the results stated in our paper.
