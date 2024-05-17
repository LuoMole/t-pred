import os

import torch
from matplotlib import pyplot as plt


#requirement1: train or test
#requirement2: sampled or not default is none


def get_dirlist(folder, requirement1, requirement2=None):
    filenamelist = []
    for filename in os.listdir(folder):
        if filename.endswith('csv') and requirement1 in filename and (requirement2 is None or requirement2 in filename):
            filenamelist.append(filename)
    return filenamelist


# data_folder = r'C:\Users\86875\Desktop\learning\t-pred\code'
# print(get_dirlist(data_folder, 'test', 'sampled'))

def evaluate_model(model_tcrep, dataloader_val, loss, device):
    model_tcrep.eval()
    total_loss = 0
    num_batches = len(dataloader_val)
    with torch.no_grad():
        for cdr3_val, ep_val, label_val in dataloader_val:
            # 将 cdr3_train 转换为张量
            cdr3_tensors_val = [torch.stack(sample, dim=0) for sample in cdr3_val]
            # 将 cdr3_tensors 转换为一个张量，添加批次维度
            cdr3_val = torch.stack(cdr3_tensors_val, dim=0).to(torch.float32)
            cdr3_val = cdr3_val.to(device)
            cdr3_val = cdr3_val.permute(2, 1, 0)

            ep_tensors_val = [torch.stack(sample, dim=0) for sample in ep_val]
            ep_val = torch.stack(ep_tensors_val, dim=0).to(torch.float32)
            ep_val = ep_val.to(device)
            ep_val = ep_val.permute(2, 1, 0)

            label_val = label_val.clone().detach()
            label_val = label_val.unsqueeze(1).to(torch.float32)
            label_val = label_val.to(device)

            output = model_tcrep(cdr3_val, ep_val)
            loss_test = loss(output, label_val)

            total_loss += loss_test
    return total_loss/num_batches

def draw_roc(fpr_list, tpr_list, mean_roc_auc_epoch, figname):
    # draw roc
    plt.Figure()
    plt.plot(fpr_list,
             tpr_list,
             color="darkorange",
             lw=2,
             label=f"ROC Curve(area={mean_roc_auc_epoch:.4f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # x,y轴范围
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    # 图例位置
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig(f'fig/{figname}')
    return

def draw_precision_recall(precision_list, recall_list, mean_roc_auc_epoch, figname):
    plt.figure()
    plt.plot(precision_list, recall_list, color="darkorange", lw=2,
             label=f"Precision_Recall Curve(area={mean_roc_auc_epoch:.4f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # x,y轴范围
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('precision')
    plt.ylabel('recall')
    plt.title('Precision_Recall')
    # 图例位置
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig(f'fig/{figname}')
    return

def draw_plot(x, y, figname, ylabel):
    plt.figure()
    plt.plot(x, y,  color="darkorange", lw=2, label=figname)
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.title('Precision_Recall')
    plt.legend(loc='lower right')
    plt.savefig(f'fig/{figname}')