import torch
import torch.nn as nn
import torch.optim as optim
from transformers import HfArgumentParser, set_seed
from config import GeolifConfig
from utils import preprocess, load_w2ifile, load_dist_dict, make_dist_index_dict
from data_provider import TrajDataloder
from model import GNN_GNN_model
from train import train
from test import test
import pickle
from pathlib import Path


if __name__ == "__main__":

    parser = HfArgumentParser(GeolifConfig)
    config = parser.parse_args_into_dataclasses()[0]
    print(config)

    set_seed(config.seed)
    config.save_path = Path(config.save_dir) / "dataset_{}_hiddensize_{}_nheads_{}_distloss_{}_dropout_{}_alpha_{}_lr_{:.4f}_nopos.chkpt".format(config.dataset, config.hidden_size, config.cross_n_heads, config.dist_loss, config.dropout_p, config.alpha, config.lr)

    w2i_dict = load_w2ifile(config.vocab_path)
    dist_dict = load_dist_dict(config.dist_path)
    dist_index_dict = make_dist_index_dict(w2i_dict, dist_dict)
    train_data_dict = preprocess(config.train_path, w2i_dict)
    valid_data_dict = preprocess(config.eval_path, w2i_dict)
    test_data_dict = preprocess(config.test_path, w2i_dict)

    device = torch.device(config.device)

    train_dataloader = TrajDataloder(train_data_dict, dist_index_dict, device, gen_pairs=config.dist_loss)
    valid_dataloader = TrajDataloder(valid_data_dict, dist_index_dict, device, gen_pairs=False)
    test_dataloader = TrajDataloder(test_data_dict, dist_index_dict, device, gen_pairs=False)

    # model definition
    model = GNN_GNN_model(config, max(w2i_dict.values())+1, train_dataloader.history_length)
    model.to(device)

    # loss function
    criterion = nn.NLLLoss().cuda()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, weight_decay=config.l2)

    # train
    train(model, optimizer, criterion, train_dataloader, config, verbose=True, validation=True, valid_dataloader=valid_dataloader, test_dataloader=test_dataloader, dist_index_dict=dist_index_dict)

    # dump the trained embedding and the w2i_dict
    dump_emb_path = Path(
        config.save_dir) / "dataset_{}_hiddensize_{}_nheads_{}_distloss_{}_dropout_{}_alpha_{}_lr_{:.4f}_nopos.emb".format(
        config.dataset, config.hidden_size, config.cross_n_heads, config.dist_loss, config.dropout_p, config.alpha,
        config.lr)
    embedding_matrix = model.embedding.weight.detach().cpu().numpy()
    pickle.dump([embedding_matrix, w2i_dict], Path(dump_emb_path).open("wb"))

    # test
    recall1, recall3, recall5, recall10, dist1, dist3, dist5, dist10, map = test(model, test_dataloader, config, dist_index_dict)
    print("recall@1: {:.4f}, recall@3: {:.4f}, recall@5: {:.4f}, recall@10: {:4f}".format(recall1, recall3, recall5, recall10))
    print("dist@1: {:.4f}, dist@3: {:.4f}, dist@5: {:.4f}, dist@10: {:4f}".format(dist1, dist3, dist5, dist10))
    print("mAP: {:.4f}".format(map))
