import torch
import numpy as np
import math
max_thresh = 1
min_thresh = 1e-7

# print(math.log(min_thresh))


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
    #  random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)



def margin_loss(cls_score, labels_one_hot, loss_weight = 0.2, min_thresh = 1e-7):
    '''
        cls_score: (n, 100/ 300, num_classes) # 不包含bg类别
        labels_one_hot: (n, 100 / 300, num_classes) # 不包含bg
    '''
    cls_prob = cls_score.sigmoid()
    target_prob, idx = (cls_prob * labels_one_hot).max(2) # n * 100 / 300
    mask = ~(labels_one_hot.bool()) # n * 100/ 300 * num_classes
    diff = (target_prob.unsqueeze(-1) - cls_prob).clamp_(min = min_thresh, max = 1)
    margin_loss = (-(diff[mask].log())).mean()  # n * 100 / 300 * (num_classes - 1), mask 之后，去除了本身的0.0
    return margin_loss


src_logits_ori = torch.rand([2, 4, 10])

target_classes = torch.tensor([
    [1,5,8,2],
    [7,4,0,9]
])
src_logits = src_logits_ori.sigmoid()
print(src_logits)
# print(target_classes)
# print(target_classes.shape)
target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
# print(target_classes_onehot)
target_classes_onehot = target_classes_onehot[:,:,:-1] # 不选取背景类
# print(target_classes_onehot)
# 

target_pred = (src_logits * target_classes_onehot)
target_pred_logits, idx = target_pred.max(2)
# print(target_pred)
# print(target_pred.max(2))
diff = target_pred_logits.unsqueeze(-1) - src_logits
mask = ~(target_classes_onehot.bool())
diff_select = diff[mask]
# print(diff_select) # 2 * 4 * 9 去掉本身
diff_select.clamp_(min = min_thresh, max = max_thresh)
loss = -diff_select.log()
# print(loss)
print(loss.mean())


print(margin_loss(src_logits_ori, target_classes_onehot))

# print(diff_select.log().reshape(8, 9))

