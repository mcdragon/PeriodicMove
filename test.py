import torch


def test(model, test_dataloader, config, dist_index_dict):
    predict_list = []
    target_list = []
    scores_list = []
    with torch.no_grad():
        model.eval()
        slices = test_dataloader.generate_batch(config.batch_size)
        for s in slices:
            batch = test_dataloader.get_slice(s)
            inputs, targets = batch[:-1], batch[-1]
            targets = targets - 3
            # batch_size * mask_num, n
            scores = model(*inputs)
            # recall@5; recall@10; distance@5; distance@10
            # batch_size * mask_num, 10
            topk_index = scores.topk(10, dim=-1, sorted=True)[1].detach().cpu().numpy().tolist()
            scores_list.extend(scores.detach().cpu().numpy().tolist())
            predict_list.extend(topk_index)
            target_list.extend(targets.detach().cpu().numpy().tolist())
    recall1, recall5, recall3, recall10, distance1, distance3, distance5, distance10, map = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for index, predict in enumerate(predict_list):
        # recall
        if target_list[index] in predict[:10]:
            recall10 += 1
        if target_list[index] in predict[:5]:
            recall5 += 1
        if target_list[index] in predict[:3]:
            recall3 += 1
        if target_list[index] in predict[:1]:
            recall1 += 1
        # distance
        dist_list = []
        for predict_point in predict:
            # here, we should remap the index to original index.
            target_point = target_list[index] + 3
            predict_point = predict_point + 3
            dist_list.append(dist_index_dict[predict_point, target_point])
        distance1 += min(dist_list[:1])
        distance3 += min(dist_list[:3])
        distance5 += min(dist_list[:5])
        distance10 += min(dist_list[:10])
        # map
        combine = sorted([(index, score) for index, score in enumerate(scores_list[index])], key=lambda x: -x[1])
        for idx, com in enumerate(combine):
            if com[0] == target_list[index]:
                map += 1.0/(idx+1)
                break
    return recall1/len(predict_list), recall3/len(predict_list), recall5/len(predict_list), recall10/len(predict_list), \
           distance1/len(predict_list), distance3/len(predict_list), distance5/len(predict_list), distance10/len(predict_list), map/len(predict_list)
