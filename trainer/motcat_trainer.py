import numpy as np
import torch

import os
from sksurv.metrics import concordance_index_censored
import random
from timeit import default_timer as timer
import sys
import time

#grade task
from sklearn.metrics import roc_auc_score,precision_score,f1_score,average_precision_score
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F
import torch.nn as nn

torch.set_num_threads(2)

def train_loop_survival_coattn_ot(epoch, bs_micro, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=32, args=None):
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c, grade) in enumerate(loader):
        #print(f"original shape: {data_WSI.shape}")
        data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
        data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
        data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
        data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
        data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
        data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
        label = label.type(torch.LongTensor).cuda()
        grade = grade.type(torch.LongTensor).cuda()
        grade = grade - 2
        c = c.type(torch.FloatTensor).cuda()

        loss = 0.
        all_risk = 0.
        cnt = 0
        
        index_chunk_list = split_chunk_list(data_WSI, bs_micro)
        #print(index_chunk_list)
        #sys.exit()
        for tindex in index_chunk_list:
            ##previous version
            
            wsi_mb = torch.index_select(data_WSI, dim=0, index=torch.LongTensor(tindex).to(data_WSI.device)).cuda()
            #print(f"wsi_mb.shape: {wsi_mb.shape}")
            hazards, S, Y_hat, A, hazard_grade = model(x_path=wsi_mb, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
            #print(hazards, hazards.shape)
            #print(label, label.shape)
            #sys.exit()
            if args.bag_loss == 'nll_surv':
                loss_micro = loss_fn(hazards=hazards, S=S, Y=label, c=c)
            elif args.bag_loss == 'cox_surv':
                loss_micro = loss_fn(hazards=hazards.squeeze(), S=S, c=c)
            else:
                raise NotImplementedError
            #loss for Deep Equilibrium Multimodal Fusion
            '''
            if jacobian_loss:
                loss_micro += args.jacobian_weight * jacobian_loss.mean()
            '''
            loss += loss_micro
            all_risk += -torch.sum(S, dim=1).detach().cpu().numpy().item()
            cnt += 1
        #sys.exit()
        loss = loss / cnt
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = all_risk / cnt # averaged risk
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time
        
        
        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 50 == 0:
            train_batch_str = 'batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}'.format(
                batch_idx, loss_value, label.item(), float(event_time), float(risk))
            with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
                f.write(train_batch_str+'\n')
            f.close()
            print(train_batch_str)
        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    c_index_train = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    train_epoch_str = 'Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(
        epoch, train_loss_surv, train_loss, c_index_train)
    print(train_epoch_str)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(train_epoch_str+'\n')
    f.close()

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index_train, epoch)

def validate_survival_coattn_ot(cur, epoch, bs_micro, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, args=None):
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    
    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    inference_times = []
    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c, grade) in enumerate(loader):
        data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
        data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
        data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
        data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
        data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
        data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
        label = label.type(torch.LongTensor).cuda()
        c = c.type(torch.FloatTensor).cuda()
        slide_id = slide_ids.iloc[batch_idx]
        
        loss = 0.
        all_risk = 0.
        cnt = 0
        with torch.no_grad():
            index_chunk_list = split_chunk_list(data_WSI, bs_micro)
            for tindex in index_chunk_list:
                wsi_mb = torch.index_select(data_WSI, dim=0, index=torch.LongTensor(tindex).to(data_WSI.device)).cuda()
                start_time = time.time()
                hazards, S, Y_hat, A, hazard_grade = model(x_path=wsi_mb, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
                end_time = time.time()
                inference_times.append(end_time - start_time)
                loss_micro = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)   
                #loss for Deep Equilibrium Multimodal Fusion
                '''
                if jacobian_loss:
                    loss_micro += args.jacobian_weight * jacobian_loss.mean()
                '''
                loss += loss_micro
                all_risk += -torch.sum(S, dim=1).detach().cpu().numpy().item()
                cnt+=1

        loss = loss / cnt
        loss_value = loss.item()
        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = all_risk / cnt # averaged risk
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(), 'survival': event_time.item(), 'censorship': c.item()}})
        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg


    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    val_epoch_str = "val c-index: {:.4f}".format(c_index)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(val_epoch_str+'\n')
    print(val_epoch_str)
    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_surv, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return patient_results, c_index, True, avg_inference_time
    avg_inference_time = np.mean(inference_times)
    return patient_results, c_index, False, avg_inference_time

def train_loop_grade_coattn_ot(epoch, bs_micro, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=32, args=None):
    model.train()
    train_loss_grade, train_loss = 0., 0.
    grad_acc_epoch = 0
    print('\n')
    #all_risk_scores = np.zeros((len(loader)))
    #all_censorships = np.zeros((len(loader)))
    #all_event_times = np.zeros((len(loader)))
    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c, grade) in enumerate(loader):
        #data_WSI = data_WSI.cuda()
        data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
        data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
        data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
        data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
        data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
        data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
        label = label.type(torch.LongTensor).cuda()
        grade = grade.type(torch.LongTensor).cuda()
        grade = grade - 2   
        c = c.type(torch.FloatTensor).cuda()
         
        loss = 0.
        #all_risk = 0.
        cnt = 0
        hazard_grade_sum = None
        '''
        hazards, S, Y_hat, A, hazard_grade  = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
        print(hazard_grade, hazard_grade.shape)
        print(grade, grade.shape)
        loss= F.nll_loss(hazard_grade, grade)
        '''
        index_chunk_list = split_chunk_list(data_WSI, bs_micro)
        #print()
        for tindex in index_chunk_list:
            wsi_mb = torch.index_select(data_WSI, dim=0, index=torch.LongTensor(tindex).to(data_WSI.device)).cuda()
            hazards, S, Y_hat, A, hazard_grade = model(x_path=wsi_mb, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
            # 累加hazard_grade
            if hazard_grade_sum is None:
                hazard_grade_sum = hazard_grade
            else:
                hazard_grade_sum += hazard_grade
            cnt += 1
            #计算损失函数
            #print(hazard_grade.shape)
            #print(grade.shape)
            loss_micro = F.nll_loss(hazard_grade, grade)
            #loss for Deep Equilibrium Multimodal Fusion
            '''
            if jacobian_loss:
                loss_micro += args.jacobian_weight * jacobian_loss.mean()
            '''
            loss += loss_micro
            
        # 计算平均hazard_grade并执行argmax
        hazard_grade_avg = hazard_grade_sum / cnt
        hazard_grade_final = hazard_grade_avg.argmax(dim=1, keepdim=True)
        
        loss = loss / cnt
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        #risk = all_risk / cnt # averaged risk
        #all_risk_scores[batch_idx] = risk
        #all_censorships[batch_idx] = c.item()
        #all_event_times[batch_idx] = event_time
        hazard_grade = hazard_grade.argmax(dim=1, keepdim=True)
        grad_acc_epoch += hazard_grade.eq(grade.view_as(hazard_grade)).sum().item()
        
        train_loss_grade += loss_value
        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 50 == 0:
            print('batch {}, loss: {:.4f}, grade_true: {}, event_time: {:.4f}, grade_pred: {:.4f}'.format(batch_idx,
                                                                                                          loss_value + loss_reg,
                                                                                                          grade.item(),
                                                                                                          float(event_time),
                                                                                                          hazard_grade.item()))
        loss = loss / gc + loss_reg
        loss.backward()
        
        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_grade /= len(loader)
    train_loss /= len(loader)
    acc = grad_acc_epoch/(len(loader))
    #c_index_train = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    print('Epoch: {}, train_loss_grade: {:.4f}, train_loss: {:.4f}, acc: {:.4f}'.format(epoch, train_loss_grade,
                                                                                                 train_loss, acc))
    #print(train_epoch_str)
    #with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
    #    f.write(train_epoch_str+'\n')
    #f.close()

    if writer:
        writer.add_scalar('train/loss_grade', train_loss_grade, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', acc, epoch)

def validate_grade_coattn_ot(cur, epoch, bs_micro, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, args=None):
    model.eval()
    '''
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    '''
    val_loss_grade=0.
    grad_acc_epoch=0
    val_loss=0.
    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    grade_true=[]
    grade_pred=[]
    grade_pred_sum=[]
    
    inference_times = []

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c, grade) in enumerate(loader):
        data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
        data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
        data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
        data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
        data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
        data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
        label = label.type(torch.LongTensor).cuda()
        c = c.type(torch.FloatTensor).cuda()
        slide_id = slide_ids.iloc[batch_idx]
        grade=grade.type(torch.LongTensor).cuda()
        grade=grade-2
        loss = 0.
        #all_risk = 0.
        cnt = 0
        hazard_grade_sum = None
        with torch.no_grad():
            index_chunk_list = split_chunk_list(data_WSI, bs_micro)
            for tindex in index_chunk_list:
                wsi_mb = torch.index_select(data_WSI, dim=0, index=torch.LongTensor(tindex).to(data_WSI.device)).cuda()
                # 计算单次推理时间
                start_time = time.time()
                hazards, S, Y_hat, A, hazard_grade = model(x_path=wsi_mb, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
                end_time = time.time()
                inference_times.append(end_time - start_time)
                # 累加hazard_grade
                if hazard_grade_sum is None:
                    hazard_grade_sum = hazard_grade
                else:
                    hazard_grade_sum += hazard_grade
                cnt += 1
                loss_micro = F.nll_loss(hazard_grade, grade)
                '''
                #loss for Deep Equilibrium Multimodal Fusion
                if jacobian_loss:
                    loss_micro += args.jacobian_weight * jacobian_loss.mean()
                '''
                loss += loss_micro

        # 计算平均hazard_grade
        hazard_grade_avg = hazard_grade_sum / cnt

        loss = loss / cnt
        loss_value = loss.item()
        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        soft=nn.Softmax(dim=1)
        hazards_grade=soft(hazard_grade_avg)
        grade_true.append(grade)
        grade_pred.append(hazards_grade)
        
        #计算分类正确率
        hazards_grade = hazards_grade.argmax(dim=1, keepdim=True) 

        grad_acc_epoch += hazards_grade.eq(grade.view_as(hazards_grade)).sum().item()
        grade_pred_sum.append(hazards_grade.item())
        
        val_loss_grade += loss_value
        val_loss += loss_value + loss_reg

    val_loss_grade /= len(loader)
    val_loss /= len(loader)
    acc=grad_acc_epoch/len(loader)
    #c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    #val_epoch_str = "val c-index: {:.4f}".format(c_index)
    #with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
    #    f.write(val_epoch_str+'\n')
    #print(val_epoch_str)
    if writer:
        writer.add_scalar('val/loss_grade', val_loss_grade, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', acc, epoch)
    try:
        grade_true=torch.cat(grade_true,dim=0).detach().cpu().numpy()
        print(grade_true,'\n')
        grade_pred=torch.cat(grade_pred,dim=0).detach().cpu().numpy()
        print(grade_pred,'\n')
        print(grade_pred_sum,'\n')
        
        grade_true_bin = label_binarize(grade_true, classes=[0, 1, 2])
        micro_auc = roc_auc_score(grade_true_bin, grade_pred,multi_class='ovr',average='micro')
        micro_ap=average_precision_score(grade_true_bin, grade_pred,average='micro')
        micro_f1=f1_score(grade_true ,grade_pred_sum,average='micro')
        acc=grad_acc_epoch/len(loader)
        print("acc:",acc,' micro_auc:',micro_auc,' micro_ap:',micro_ap,' micro_f1:',micro_f1,'\n')
    except ValueError as V:
        print(V)   		    
    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_grade, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            patient_results, acc, micro_auc, micro_ap, micro_f1, True, avg_inference_time
    avg_inference_time = np.mean(inference_times)        
    return patient_results, acc, micro_auc, micro_ap, micro_f1, False, avg_inference_time


def split_chunk_list(data, batch_size):
    numGroup = data.shape[0] // batch_size + 1
    feat_index = list(range(data.shape[0]))
    random.shuffle(feat_index)
    index_chunk_list = np.array_split(np.array(feat_index), numGroup)
    index_chunk_list = [sst.tolist() for sst in index_chunk_list]
    #for t in index_chunk_list: print(len(t))
    #sys.exit()
    return index_chunk_list
