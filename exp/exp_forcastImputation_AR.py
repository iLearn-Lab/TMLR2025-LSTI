from data_provider.data_factory_my import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping_my, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')


class Exp_ForecastImputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_ForecastImputation, self).__init__(args)
        self.model_back = self._build_model().to(self.device)
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.AR_len = args.AR_len
        self.HaveCorrelationLoss = False


    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_optimizer_back(self):
        model_optim = optim.Adam(self.model_back.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        self.model_back.eval()

        # 正向预测
        with torch.no_grad():
            data_miss = vali_data.data_miss.copy()
            mask_tot = np.isnan(data_miss)
            true = vali_data.data_x[mask_tot].copy()
            true = np.nan_to_num(true)
            for i in range(self.args.seq_len, data_miss.shape[0] - self.pred_len, self.pred_len):
                mask = np.isnan(data_miss[i : i + self.pred_len, :])
                if np.any(mask):
                    series = data_miss[i-self.args.seq_len:i, :]
                    series = torch.tensor(series).float().to(self.device).unsqueeze(0)
                    series_mark = vali_data.data_stamp[i-self.args.seq_len:i, :]
                    series_mark = torch.tensor(series_mark).float().to(self.device).unsqueeze(0)
                    
                    outputs = self.model(series, series_mark, None, None)
                    outputs = outputs.squeeze(0).cpu().numpy()
                    outputs[~mask] = (data_miss[i : i + self.pred_len, :])[~mask]

                    data_miss[i : i + self.pred_len, :] = outputs
            
            pred_front = data_miss.copy()
            
        # 反向预测
        with torch.no_grad():
            data_miss_b = vali_data.data_miss
            data_miss_b = data_miss_b[::-1, :].copy()
            data_stamp = vali_data.data_stamp
            data_stamp = data_stamp[::-1, :].copy()
            for i in range(self.args.seq_len, data_miss_b.shape[0] - self.pred_len, self.pred_len):
                mask_b = np.isnan(data_miss_b[i : i + self.pred_len, :])
                if np.any(mask_b):
                    series_b = data_miss_b[i-self.args.seq_len:i, :]
                    series_b = torch.tensor(series_b).float().to(self.device).unsqueeze(0)
                    series_mark_b = data_stamp[i-self.args.seq_len:i, :]
                    series_mark_b = torch.tensor(series_mark_b).float().to(self.device).unsqueeze(0)
                    outputs = self.model_back(series_b, series_mark_b, None, None)
                    outputs = outputs.squeeze(0).cpu().numpy()
                    outputs[~mask_b] = (data_miss_b[i : i + self.pred_len, :])[~mask_b]

                    data_miss_b[i : i + self.pred_len, :] = outputs
            
            pred_back = data_miss_b[::-1, :].copy()


        # 扫描整个序列，对于每个空缺的区间，进行前后的加权融合
        l=r=0
        pred = (pred_front + pred_back) / 2
        while(l < len(pred_front) and r < len(pred_front)):
            mask = np.isnan(vali_data.data_miss[r, :])
            if np.any(mask) and r < len(pred_front) - 1: 
                r = r + 1
            else:
                if l == r:
                    l = l + 1
                    r = r + 1
                else:
                    miss_len = r - l
                    weight_f = np.arange(miss_len - 1, -1, -1).astype(float) / (miss_len - 1)
                    weight_b = np.arange(miss_len).astype(float) / (miss_len - 1)
                    pred[l : r] = (pred_front[l : r] * weight_f.reshape(miss_len, 1) + pred_back[l : r] * weight_b.reshape(miss_len, 1))
                    r = r + 1
                    l = r

        pred = pred[mask_tot]
        pred_front = pred_front[mask_tot]
        pred_back = pred_back[mask_tot]

        # seq_len = len(pred_front)
        # weight_f = np.arange(seq_len - 1, -1, -1).astype(float) / (seq_len - 1)
        # weight_b = np.arange(seq_len).astype(float) / (seq_len - 1)
        # pred = (pred_front * weight_f + pred_back * weight_b)

        loss = mean_squared_error(pred, true)
        loss_front = mean_squared_error(pred_front, true)
        loss_back = mean_squared_error(pred_back, true)
        print('loss: {}  loss_front: {}  loss_back: {}'.format(loss, loss_front, loss_back))
                        
        total_loss = torch.tensor(loss)
        self.model.train()
        self.model_back.train()
        return total_loss


    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        path_back = os.path.join(self.args.checkpoints, setting, 'back')
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path_back):
            os.makedirs(path_back)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping_my(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        model_optim_back = self._select_optimizer_back()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_loss_f = []
            train_loss_b = []

            self.model.train()
            self.model_back.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,batch_z, batch_z_mark) in enumerate(train_loader):
                iter_count += 1
                # model_optim.zero_grad()
                # model_optim_back.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_z = batch_z.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_z_mark = batch_z_mark.float().to(self.device)

                # 自回归预测：
                inputs_front = batch_x.clone()
                stamp_front = torch.cat((batch_x_mark, batch_y_mark), dim=1)
                inputs_back = batch_z.clone()
                stamp_back = torch.cat((batch_z_mark, batch_y_mark.flip(dims=[1])), dim=1)
                
                for j in range(0, self.AR_len * self.pred_len, self.pred_len):
                    model_optim.zero_grad()
                    model_optim_back.zero_grad()
                    
                    if j > 0:
                        inputs_front = torch.cat([inputs_front[:, self.pred_len:, :], outputs_f.detach()], dim=1)
                        inputs_back = torch.cat([inputs_back[:, self.pred_len:, :], outputs_b.detach()], dim=1)
                    series_mark_front = stamp_front[:, j:j+self.seq_len, :]
                    series_mark_back = stamp_back[:, j:j+self.seq_len, :]

                    outputs_f = self.model(inputs_front, series_mark_front, None, None)
                    outputs_b = self.model_back(inputs_back, series_mark_back, None, None)

                    f_id = j
                    b_id = (self.AR_len-1) * self.pred_len - j
                    true_f = batch_y[:, f_id : f_id + self.pred_len, :]
                    true_b = batch_y[:, b_id : b_id + self.pred_len, :].flip(dims=[1])
                    loss_f = criterion(outputs_f, true_f)
                    loss_b = criterion(outputs_b, true_b)
                    if f_id == b_id:
                        loss = loss_f + loss_b + criterion(outputs_f, outputs_b.flip(dims=[1]))
                    else:
                        loss = loss_f + loss_b
                    # loss = loss_f + loss_b
                        
                    loss.backward()
                    model_optim.step()
                    model_optim_back.step()
                    train_loss.append(loss.item())
                    train_loss_f.append(loss_f.item())
                    train_loss_b.append(loss_b.item())

                    if (iter_count + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}  loss_f:{3:.7f}  loss_b:{4:.7f}".format(iter_count + 1, epoch + 1, loss.item(), loss_f.item(), loss_b.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            # self.test(setting)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}  Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, self.model_back, path, path_back)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            adjust_learning_rate(model_optim_back, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        best_model_path_back = path_back + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.model_back.load_state_dict(torch.load(best_model_path_back))

        return self.model, self.model_back

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.model_back.eval()
        # 正向预测
        with torch.no_grad():
            data_miss = test_data.data_miss.copy()
            mask_tot = np.isnan(data_miss)
            true = test_data.data_x[mask_tot].copy()
            true = np.nan_to_num(true)
            for i in range(self.args.seq_len, data_miss.shape[0] - self.pred_len, self.pred_len):
                mask = np.isnan(data_miss[i : i + self.pred_len, :])
                if np.any(mask):
                    series = data_miss[i-self.args.seq_len:i, :]
                    series = torch.tensor(series).float().to(self.device).unsqueeze(0)
                    series_mark = test_data.data_stamp[i-self.args.seq_len:i, :]
                    series_mark = torch.tensor(series_mark).float().to(self.device).unsqueeze(0)
                    
                    outputs = self.model(series, series_mark, None, None)
                    outputs = outputs.squeeze(0).cpu().numpy()
                    outputs[~mask] = (data_miss[i : i + self.pred_len, :])[~mask]

                    data_miss[i : i + self.pred_len, :] = outputs
            
            pred_front = data_miss.copy()
            
        # 反向预测
        with torch.no_grad():
            data_miss_b = test_data.data_miss
            data_miss_b = data_miss_b[::-1, :].copy()
            data_stamp = test_data.data_stamp
            data_stamp = data_stamp[::-1, :].copy()
            for i in range(self.args.seq_len, data_miss_b.shape[0] - self.pred_len, self.pred_len):
                mask_b = np.isnan(data_miss_b[i : i + self.pred_len, :])
                if np.any(mask_b):
                    series_b = data_miss_b[i-self.args.seq_len:i, :]
                    series_b = torch.tensor(series_b).float().to(self.device).unsqueeze(0)
                    series_mark_b = data_stamp[i-self.args.seq_len:i, :]
                    series_mark_b = torch.tensor(series_mark_b).float().to(self.device).unsqueeze(0)
                    outputs = self.model_back(series_b, series_mark_b, None, None)
                    outputs = outputs.squeeze(0).cpu().numpy()
                    outputs[~mask_b] = (data_miss_b[i : i + self.pred_len, :])[~mask_b]

                    data_miss_b[i : i + self.pred_len, :] = outputs
            
            pred_back = data_miss_b[::-1, :].copy()

        # 扫描整个序列，对于每个空缺的区间，进行前后的加权融合。同时打印图像
        l=r=0
        pred = (pred_front + pred_back) / 2
        while(l < len(pred_front) and r < len(pred_front)):
            mask = np.isnan(test_data.data_miss[r, :])
            if np.any(mask) and r < len(pred_front) - 1: 
                r = r + 1
            else:
                if l == r:
                    l = l + 1
                    r = r + 1
                else:
                    miss_len = r - l
                    weight_f = np.arange(miss_len - 1, -1, -1).astype(float) / (miss_len - 1)
                    weight_b = np.arange(miss_len).astype(float) / (miss_len - 1)
                    pred[l : r] = (pred_front[l : r] * weight_f.reshape(miss_len, 1) + pred_back[l : r] * weight_b.reshape(miss_len, 1))
                    # for c in range(pred.shape[1]):
                    #     visual(test_data.data_x[l:r, c], pred_front[l:r, c], os.path.join(folder_path, str(r) + "_" + str(c) + '_front.png'))
                    #     visual(test_data.data_x[l:r, c], pred_back[l:r, c], os.path.join(folder_path, str(r) + "_" + str(c) + '_back.png'))
                    #     visual(test_data.data_x[l:r, c], pred[l:r, c], os.path.join(folder_path, str(r) + "_" + str(c) + '_pred.png'))
                    r = r + 1
                    l = r

        pred = pred[mask_tot]
        pred_front = pred_front[mask_tot]
        pred_back = pred_back[mask_tot]

        # seq_len = len(pred_front)
        # weight_f = np.arange(seq_len - 1, -1, -1).astype(float) / (seq_len - 1)
        # weight_b = np.arange(seq_len).astype(float) / (seq_len - 1)
        # pred = (pred_front * weight_f + pred_back * weight_b)

        mae_f = mean_absolute_error(pred_front, true)
        mae_b = mean_absolute_error(pred_back, true)
        mse_f = mean_squared_error(pred_front, true)
        mse_b = mean_squared_error(pred_back, true)
        mae = mean_absolute_error(pred, true)
        mse = mean_squared_error(pred, true)

        print("test")
        print("mse_f_b:", mse_f, mse_b)
        print("mae_f_b:", mae_f, mae_b)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_forecastImputation.txt", 'a')
        f.write(setting + "  \n")
        f.write('seed:{}, rate:{}'.format(self.args.seed, self.args.mask_rate) + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()
        return mse
