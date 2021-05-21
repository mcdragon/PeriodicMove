import torch
from earlystop import EarlyStopping
from test import test


def fn_dist_loss(model, anchor_points, pos_points, neg_points, pos_weights):
    anchor_emb = model.get_embedding(anchor_points)
    pos_emb = model.get_embedding(pos_points)
    neg_emb = model.get_embedding(neg_points)
    triple_loss = torch.nn.functional.triplet_margin_loss(anchor_emb, pos_emb, neg_emb, reduction="none")
    dist_loss = (triple_loss * pos_weights).sum()
    return dist_loss


def train(model, optimizer, criterion, train_dataloader, config, verbose=True, validation=True, valid_dataloader=None, test_dataloader=None, dist_index_dict=None):

    early_stopping = EarlyStopping(config.patience, verbose=True, save_path=config.save_path, reverse=False)

    alpha = config.alpha
    alpha_decay = 0.8
    for epoch in range(config.epochs):
        # training
        model.train()
        training_epoch_loss, train_recover_loss, train_dist_loss = 0.0, 0.0, 0.0
        slices = train_dataloader.generate_batch(config.batch_size)
        for s in slices:
            optimizer.zero_grad()
            if config.dist_loss:
                batch = train_dataloader.get_slice(s)
                inputs, targets, anchor_points, pos_points, neg_points, pos_weights = batch[:-5], batch[-5], \
                                                                                      batch[-4], batch[-3], batch[-2], batch[-1]
                scores = model(*inputs)
                recover_loss = criterion(scores, targets-3)
                if epoch < 1000:
                    dist_loss = fn_dist_loss(model, anchor_points, pos_points, neg_points, pos_weights)
                    loss = recover_loss + alpha * dist_loss
                    train_dist_loss += alpha * dist_loss
                else:
                    loss = recover_loss
                train_recover_loss += recover_loss.item()
            else:
                batch = train_dataloader.get_slice(s)
                inputs, targets = batch[:-1], batch[-1]
                scores = model(*inputs)
                loss = criterion(scores, targets-3)
            loss.backward()
            optimizer.step()
            # todo: grad_norm
            training_epoch_loss += loss.item()
        alpha = alpha * alpha_decay
        if verbose == True:
            print("epoch: {}, training loss: {:.4f}, recover loss: {:.4f}, dist loss: {:4f}".format(epoch, training_epoch_loss, train_recover_loss, train_dist_loss))

        # validation
        if validation and valid_dataloader is not None:
            recall1, _, _, _, _, _, _, _, _ = test(model, valid_dataloader, config, dist_index_dict)
            early_stopping(recall1, model)
            if early_stopping.early_stop:
                print("Early Stopping!")
                break

    # load the trained model
    if validation and valid_dataloader is not None:
        model.load_state_dict(torch.load(config.save_path))

