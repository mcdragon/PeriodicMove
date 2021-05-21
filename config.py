from dataclasses import dataclass, field


@dataclass
class GeolifConfig:
    # reproducing configuration
    seed: int = field(metadata={"help": "to reproducing the results in the paper."}, default=2021)

    dataset: str = field(metadata={"help": "string of dataset"}, default="foursquare")
    vocab_path: str = field(metadata={"help": "the path of vocab file"},
                           default="../datasets/processed/foursquare/v3_delta_0.005_window_size_6_mask_num_10/pos.vocab.txt")
    dist_path: str = field(metadata={"help": "the path of distance file"},
                          default="../datasets/processed/foursquare/v3_delta_0.005_window_size_6_mask_num_10/vocabs_dist.txt")
    train_path: str = field(metadata={"help": "the path of training data"},
                           default="../datasets/processed/foursquare/v3_delta_0.005_window_size_6_mask_num_10/pos.train.txt")
    eval_path: str = field(metadata={"help": "the path of validation data"},
                          default="../datasets/processed/foursquare/v3_delta_0.005_window_size_6_mask_num_10/pos.validate.txt")
    test_path: str = field(metadata={"help": "the path of testing data"},
                          default="../datasets/processed/foursquare/v3_delta_0.005_window_size_6_mask_num_10/pos.test.txt")
    save_dir: str = field(metadata={"help": "the path for saving model"}, default="./save_models/")
    dump_emb_path: str = field(metadata={"help": "the path for saving embedding"},
                              default="../datasets/processed/foursquare/v3_delta_0.005_window_size_6_mask_num_10/emb_w2i.pkl")

    # training configuration
    device: str = field(metadata={"help": "the running device"}, default="cuda:2")
    epochs: int = field(metadata={"help": "training epochs"}, default=0)
    batch_size: int = field(metadata={"help": "the training/validation/testing batch size"}, default=50)
    dropout_p: float = field(metadata={"help": "dropout rate"}, default=0.3)
    step: int = field(metadata={"help": "the steps of GGNN"}, default=1)
    lr: float = field(metadata={"help": "learning rate"}, default=1e-3)
    l2: float = field(metadata={"help": "weight decay"}, default=1e-5)
    patience: float = field(metadata={"help": "patience for early stopping"}, default=10)
    dist_loss: bool = field(metadata={"help": "add distance loss"}, default=True)
    alpha: float = field(metadata={"help": "loss balance weight"}, default=0.10)

    # model configuration
    hidden_size: int = field(metadata={"help": "hidden size of the model"}, default=128)
    cross_n_heads: int = field(metadata={"help": "num of heads in cross attention layer"}, default=4)
