import random
import numpy as np
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader
import time
from torch.distributions import Categorical
# import matplotlib.pyplot as plt
import gc
# import seaborn as sns
import pandas as pd

from torchfm.dataset.avazu import AvazuDataset
from torchfm.dataset.criteo import CriteoDataset
from torchfm.dataset.movielens import Movielens1MDataset, MovieLens20MDataset
from torchfm.dataset.kuaiRec import KuaiRecDataset
from torchfm.model.afi import AutomaticFeatureInteractionModel
from torchfm.model.afm import AttentionalFactorizationMachineModel
from torchfm.model.dcn import DeepCrossNetworkModel
from torchfm.model.dfm import DeepFactorizationMachineModel
from torchfm.model.ffm import FieldAwareFactorizationMachineModel
from torchfm.model.fm import FactorizationMachineModel
from torchfm.model.fnfm import FieldAwareNeuralFactorizationMachineModel
from torchfm.model.fnn import FactorizationSupportedNeuralNetworkModel
from torchfm.model.hofm import HighOrderFactorizationMachineModel
from torchfm.model.lr import LogisticRegressionModel
from torchfm.model.ncf import NeuralCollaborativeFiltering
from torchfm.model.nfm import NeuralFactorizationMachineModel
from torchfm.model.pnn import ProductNeuralNetworkModel
from torchfm.model.wd import WideAndDeepModel
from torchfm.model.xdfm import ExtremeDeepFactorizationMachineModel
from torchfm.model.afn import AdaptiveFactorizationNetwork
from torchfm.network import kmax_pooling
from selected_data.data_process import SelectedDataset, save_selected_data, clear_selected_data, backup_best_data, record_excel


class ControllerNetwork_instance(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

        self.controller_losses = None
    
    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        output_layer = self.mlp(embed_x.view(-1, self.embed_output_dim))
        return output_layer#torch.softmax(output_layer, dim=0).squeeze()

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 2))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

def set_random_seed(seed):
    print("* random_seed:", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_dataset(name, path):
    if name == 'selected_data':
        return SelectedDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, field_dims):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    # field_dims = dataset.field_dims
    if name == 'lr':
        return LogisticRegressionModel(field_dims)
    elif name == 'fm':
        return FactorizationMachineModel(field_dims, embed_dim=16)
    elif name == 'hofm':
        return HighOrderFactorizationMachineModel(field_dims, order=3, embed_dim=16)
    elif name == 'ffm':
        return FieldAwareFactorizationMachineModel(field_dims, embed_dim=4)
    elif name == 'fnn':
        return FactorizationSupportedNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'wd':
        return WideAndDeepModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'ipnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16,), method='inner', dropout=0.2)
    elif name == 'opnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16,), method='outer', dropout=0.2)
    elif name == 'dcn':
        return DeepCrossNetworkModel(field_dims, embed_dim=16, num_layers=3, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'nfm':
        return NeuralFactorizationMachineModel(field_dims, embed_dim=64, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'fnfm':
        return FieldAwareNeuralFactorizationMachineModel(field_dims, embed_dim=4, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'dfm':
        return DeepFactorizationMachineModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'xdfm':
        return ExtremeDeepFactorizationMachineModel(
            field_dims, embed_dim=16, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'afm':
        return AttentionalFactorizationMachineModel(field_dims, embed_dim=16, attn_size=16, dropouts=(0.2, 0.2))
    elif name == 'afi':
        return AutomaticFeatureInteractionModel(
            field_dims, embed_dim=16, atten_embed_dim=64, num_heads=2, num_layers=3, mlp_dims=(400, 400), dropouts=(0, 0, 0))
    elif name == 'afn':
        print("Model:AFN")
        return AdaptiveFactorizationNetwork(
            field_dims, embed_dim=16, LNN_dim=1500, mlp_dims=(400, 400, 400), dropouts=(0, 0, 0))
    else:
        raise ValueError('unknown model name: ' + name)


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def train_pre(model, optimizer, data_loader, criterion, device, losses, training_step, log_interval=100):
    model.train()
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        if training_step[0] == 1:  # retrain
            fields, target = fields.long(), target.long()
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss_list = criterion(y, target.float())
        if training_step[0] == 0:  # pretrain
            batch_loss = loss_list.detach()
            losses[training_step[1]].extend(batch_loss)
        loss = loss_list.mean()
        model.zero_grad()
        loss.backward()
        optimizer.step()


def test_pre(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts), log_loss(targets, predicts)


def train_noFullBatch(model, controller, optimizer, optimizer_controller, data_loader, criterion, device, batch_size, data_a_batch, losses, selected_data_path, epoch_i, dataset_name, ControllerLoss, epsilon):
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    prev_avg_list = torch.sum(losses[:-1, :], dim=0).div((len(losses)-1))
    idx_start = 0
    idx_end = 0

    rewards = []

    for i, (fields, target) in enumerate(tk0):

        idx_end = idx_end + len(target)
        fields, target = fields.to(device), target.to(device)

        repeats = 3
        controller.train()
        for _ in range(repeats):
            output_layer = controller(fields)
            prob_instance = torch.softmax(output_layer, dim=-1)

            # sample data_a_batch times from prob_instance distribution, no repeat selection
            try:
                sampled_actions = torch.argmax(prob_instance, dim=1).squeeze()
                sampled_actions = torch.tensor([action if random.random()>epsilon else -(action-1) for action in sampled_actions]).to(device)
                prob_idx = torch.nonzero(sampled_actions).squeeze()
            except:
                print("error: ================================", i)
                continue
        
            # get the selected instance
            selected_target = torch.gather(target, 0, prob_idx)
            selected_instance = torch.gather(
                fields, 0, prob_idx.unsqueeze(1).repeat(1, fields.shape[1]))

            model.eval()
            with torch.no_grad():
                y = model(selected_instance)
                loss_list = criterion(y, selected_target.float())

                # batch_loss = loss_list.detach()
                avg_list = torch.Tensor([prev_avg_list[idx]
                                        for idx in prob_idx.add(idx_start)]).to(device)
                reward_list = torch.sub(avg_list, loss_list)

                rewards.append(torch.sum(reward_list).cpu())

            c_loss = torch.sum(ControllerLoss(output_layer, sampled_actions) * torch.sum(reward_list))
            
            controller.zero_grad()
            c_loss.backward()
            optimizer_controller.step()

        save_update_model(controller, model, fields, target, device,
                          selected_data_path, criterion, optimizer, losses, idx_start, prev_avg_list)
        
        idx_start = idx_end
        


def save_update_model(controller, model, fields, target, device, selected_data_path, criterion, optimizer, losses, idx_start, prev_avg_list):
    controller.eval()
    with torch.no_grad():
        output_layer = controller(fields)
        prob_instance = torch.softmax(output_layer, dim=-1)
        try:
            sampled_actions = torch.argmax(prob_instance, dim=-1).squeeze()
            prob_idx = torch.nonzero(sampled_actions).squeeze()
        except:
            print("\t error: kmax_pooling fail!")
            return
        selected_target = torch.gather(target, 0, prob_idx)
        selected_instance = torch.gather(
            fields, 0, prob_idx.unsqueeze(1).repeat(1, fields.shape[1]))
    
    model.train()
    y = model(selected_instance)
    loss_list = criterion(y, selected_target.float())
    avg_list = torch.Tensor([prev_avg_list[idx]
                                        for idx in prob_idx.add(idx_start)]).to(device)
    reward_batch = torch.sum(torch.sub(avg_list, loss_list))
    print(f'\t test reward: {reward_batch}') 
    loss = loss_list.mean()
    model.zero_grad()
    loss.backward()
    optimizer.step()
    
    added_idx = prob_idx.add(idx_start)
    for ii in range(len(prob_idx)):
        losses[-1][added_idx[ii]] = loss_list[ii]
    


def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts), log_loss(targets, predicts)


def train_re( field_dims,train_data_loader, valid_data_loader, test_data_loader,
             model_name, learning_rate, criterion, weight_decay, device, training_step, save_model_name):
    model = get_model(model_name, field_dims).to(device)
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(
        num_trials=2, save_path=save_model_name.replace('controller', 'model')+'.pt')
    training_step[0] = 1
    train_start_time = time.time()
    for epoch_i in range(100):
        train_pre(model, optimizer, train_data_loader,
                  criterion, device, [], training_step)
        auc, logloss = test_pre(model, valid_data_loader, device)
        print('\tretrain epoch:', epoch_i,
              'validation: auc:', auc, 'logloss:', logloss)
        if not early_stopper.is_continuable(model, auc):
            print(f'\tvalidation: best auc: {early_stopper.best_accuracy}')
            break
    train_end_time = time.time()
    print("\tTime of retrain: {:.2f}min".format(
        (train_end_time - train_start_time)/60))
    auc, logloss = test_pre(model, test_data_loader, device)
    print(f'\tretrain test auc: {auc}, logloss: {logloss}\n')
    return auc, logloss


def save_test_validset(data_loader, selected_data_path):
    clear_selected_data(selected_data_path)
    print('Start saving', selected_data_path)
    for (fields, target) in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
        save_selected_data(selected_data_path, fields.cpu(
        ).numpy().copy(), target.cpu().numpy().copy())
    print('Finish saving.')

def select_instance(data_loader, selected_data_path, controller, batch_size, data_a_batch, device, select_ratio):
    controller.eval()
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    clear_selected_data(selected_data_path)
    slct_number = 0
    for i, (fields, target) in enumerate(tk0):
        if len(target) < batch_size:
            data_a_batch = int(round(select_ratio * len(target)))
        fields, target = fields.to(device), target.to(device)
        with torch.no_grad():
            output_layer = controller(fields)
            prob_instance = torch.softmax(output_layer, dim=-1)
            try:
                prob_idx = kmax_pooling(prob_instance[:,1], 0, data_a_batch)
            except:
                print("\t error: select_instance fail!")
                return
            selected_target = torch.gather(target, 0, prob_idx)
            selected_instance = torch.gather(
                fields, 0, prob_idx.unsqueeze(1).repeat(1, fields.shape[1]))
            save_selected_data(selected_data_path, selected_instance.cpu().numpy(), selected_target.cpu().numpy())
            slct_number += selected_target.shape[0]
    return slct_number

def main(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir,
         select_ratio,
         pretrain_epoch,
         retrain_per_n,
         epsilon):
    print('Dataset:', dataset_name)
    print('Dataset path:', dataset_path)
    data_a_batch = int(round(select_ratio * batch_size))
    criterion = torch.nn.BCELoss(reduction='none')
    info = '{}_{}_{}_{}_{}_{}_{}'.format(model_name, dataset_name, str(
        pretrain_epoch), str(epoch), str(batch_size), str(data_a_batch), str(select_ratio))
    save_model_name = './{}/controller_whole_'.format(
        save_dir) + info
    selected_data_path = './selected_data/notFixed_whole_{}_{}_{}_{}_train.txt'.format(
        dataset_name, pretrain_epoch, epsilon, select_ratio*100)
    print('Training batch size:', batch_size)
    print('Size of selected data in a training batch:',
          data_a_batch, select_ratio)
    print('epsilon:', epsilon)

    device = torch.device(device)

    path_train = './data/'+dataset_name+'_train_12345.txt'
    path_val = './data/'+dataset_name+'_valid_12345.txt'
    path_test = './data/'+dataset_name+'_test_12345.txt'
    print("-original train:", path_train)
    print("-original valid:", path_val)
    print("-original test:", path_test)
    dataset_train = get_dataset('selected_data', path_train)
    dataset_valid = get_dataset('selected_data', path_val)
    dataset_test = get_dataset('selected_data', path_test)

    train_data_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(dataset_valid, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=8)

    field_dims = []
    for i in range(len(dataset_train.field_dims)):
        field_dims.append(max(dataset_train.field_dims[i], dataset_valid.field_dims[i], dataset_test.field_dims[i]))

    # controller
    controller = ControllerNetwork_instance(
        field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2).to(device)
    optimizer_controller = torch.optim.Adam(
        params=controller.parameters(), lr=learning_rate*10, weight_decay=weight_decay*10)
    ControllerLoss = nn.CrossEntropyLoss(reduction='none')

    # last row for the current epoch
    losses = [[] for _ in range(pretrain_epoch+1)]
    print("losses matrix before:", losses, len(losses))
    print('\n********************************************* Pretrain *********************************************\n')
    # [0, 0] for pretrain/retrain and loss_covered(epoch_i)
    training_step = [0]*2
    train_start_time = time.time()
    sumAUC, sumLogloss = 0, 0
    for epoch_i in range(pretrain_epoch):
        model = get_model(model_name, field_dims).to(device)

        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        print('Pretrain epoch:', epoch_i)
        training_step[1] = epoch_i
        train_pre(model, optimizer, train_data_loader,
                  criterion, device, losses, training_step)
        auc, logloss = test_pre(model, test_data_loader, device)
        print(f'Pretrain epoch: {epoch_i} test auc: {auc} logloss: {logloss}')
        sumAUC, sumLogloss = sumAUC+auc, sumLogloss+logloss
    train_end_time = time.time()
    print("Time of pretrain: {:.2f}min".format(
        (train_end_time - train_start_time)/60))
    print("Average performance pretrain: auc_{:.8f}  logloss_{:.8f}".format(
        sumAUC/pretrain_epoch, sumLogloss/pretrain_epoch))

    # last row for the current epoch
    [losses[-1].append(1) for _ in range(len(losses[0]))]
    print("losses matrix:", "row_num:", len(
        losses), "colunm_num:", len(losses[0]))
    print("Transforming losses matrix...")
    t0 = time.time()
    losses = torch.Tensor(losses).to(device)
    t1 = time.time()
    print("Transform Time:", t1-t0)

    print('\n********************************************* train *********************************************\n')
    train_start_time = time.time()
    sumAUC, sumLogloss = 0, 0
    retrain_performance = []
    maxAUC_retrain = 0
    for epoch_i in range(epoch):
        model = get_model(model_name, field_dims).to(device)
        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        training_step[1] = epoch_i

        train_noFullBatch(model, controller, optimizer, optimizer_controller, train_data_loader,
                              criterion, device, batch_size, data_a_batch, losses, selected_data_path, epoch_i, dataset_name, ControllerLoss, epsilon)
        auc, logloss = test(model, test_data_loader, device)
        print(f'Train epoch: {epoch_i} test auc: {auc} logloss: {logloss}')
        sumAUC, sumLogloss = sumAUC+auc, sumLogloss+logloss
        training_step[1] %= len(losses)-1
        print("losses row covered:", training_step[1], "/", len(losses)-2)
        losses[training_step[1]] = losses[-1].clone()
    
        print(
            '\n\t========================= retrain start ==========================\n')
        print('\t# start selection...')
        slct_number = select_instance(train_data_loader, selected_data_path, controller, batch_size, data_a_batch, device, select_ratio)
        print('\t# end selection...\t total number:', slct_number, '\n\tpath:', selected_data_path)
        selected_data_loader = DataLoader(SelectedDataset(
            selected_data_path), batch_size=batch_size, shuffle=False)
        retrain_auc, retrain_logloss = train_re(field_dims, selected_data_loader, valid_data_loader, test_data_loader,
                                                model_name, learning_rate, criterion, weight_decay, device, training_step, save_model_name)
        retrain_performance.append([retrain_auc, retrain_logloss])
        # Save controller and selected instances
        if (retrain_auc > maxAUC_retrain):  # and (pretrain_epoch == 3):
            torch.save(controller, save_model_name + '.pt')
            print('\tTrained controller saved to ' + save_model_name + '.pt')
            backup_best_data(selected_data_path)
            print('\tBest data saved to ' +
                    selected_data_path.replace('train', 'best'))
            maxAUC_retrain = retrain_auc
        print('\t========================= retrain end ===========================\n')
        print("--------------------------------------------------------------------------------------------------------")
    train_end_time = time.time()
    print("Time of training: {:.2f}min".format(
        (train_end_time - train_start_time)/60))
    print("Average performance train: auc_{:.6f}  logloss_{:.6f}".format(
        sumAUC/epoch, sumLogloss/epoch))

    with open('Record_data/%s_%s_notFixed_whole.txt' % (model_name, dataset_name), 'a') as the_file:
        the_file.write('\nModel:%s\nDataset:%s\ntrain Time:%.2f,train Epoches: %d, batch_size: %d, epsilon: %s\n retrain performance: %s\n'
                       % (model_name, dataset_name, (train_end_time - train_start_time)/60, epoch_i+1, batch_size, str(epsilon), str(retrain_performance)))


if __name__ == '__main__':
    set_random_seed(12345)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='movielens1M')
    parser.add_argument(
        '--dataset_path', help='criteo/train.txt, avazu/train, or ml-1m/ratings.dat', default='')
    parser.add_argument('--model_name', default='dfm')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--pretrain_epoch', type=int, default=4)
    parser.add_argument('--retrain_per_n', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument(
        '--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', default='save_model')
    parser.add_argument('--repeat_experiments', type=int, default=1)
    parser.add_argument('--select_ratio',  type=float, default=0.98)
    parser.add_argument('--epsilon', type=float, default=0.2)
    args = parser.parse_args()

    slct_number = 0

    for i in range(args.repeat_experiments):
        main(args.dataset_name,
             args.dataset_path,
             args.model_name,
             args.epoch,
             args.learning_rate,
             args.batch_size,
             args.weight_decay,
             args.device,
             args.save_dir,
             args.select_ratio,
             args.pretrain_epoch,
             args.retrain_per_n,
             args.epsilon)
