#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import torch.nn as nn
from RBF_Layer import RBFLayer
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau



class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader, cfg, use_rbf=False, optimizer=None, criterion=None, patience=100, save_path=None):
        # Initialize the trainer with the model, data, configuration, and training options
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.cfg = cfg
        
        self.use_rbf = use_rbf  # Flag to indicate whether to use the RBF layer

        # Set up the optimizer and loss function
        self.optimizer = optimizer if optimizer else optim.AdamW(model.parameters(), lr=self.cfg.model.lr, weight_decay=self.cfg.model.weight_decay)
        # Define the scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        self.criterion = criterion if criterion else nn.MSELoss()

        # 【毕设修改 M2】RBF 参数正则系数。三个值默认 0 / -2.0，
        # 即不往损失里加任何额外项，原训练行为完全不变。
        self.center_div_lambda = float(self.cfg.model.get("center_div_lambda", 0.0) or 0.0)
        self.gamma_reg_lambda = float(self.cfg.model.get("gamma_reg_lambda", 0.0) or 0.0)
        self.gamma_reg_target = float(self.cfg.model.get("gamma_reg_target", -2.0))

        # 【毕设修改 M5】密度对齐损失系数与目标值。
        # lambda=0（默认）时不参与损失，原训练行为不变。
        self.density_align_lambda = float(self.cfg.model.get("density_align_lambda", 0.0) or 0.0)
        self.density_align_target = float(self.cfg.model.get("density_align_target", 0.9))
        self.last_reg_details = {"center_div": 0.0, "gamma_reg": 0.0, "density": 0.0}

        # Set up early stopping
        self.best_loss = float('inf')
        self.patience = patience
        # 【并行修复】暂存最优权重的文件名必须带上进程号。
        # 原作者用一个固定的 'best_model_early.pth'，而所有进程的工作目录都是 restad/，
        # 于是多进程并发时会出现：A 进程刚写完、B 进程把它删掉、A 再 load 时直接
        # EOFError: Ran out of input。加上 pid 后各进程互不干扰。
        # 单进程跑时只是文件名多了个后缀，行为与原来完全一致。
        self.save_path = save_path if save_path is not None else f"best_model_early_{os.getpid()}.pth"
        self.counter = 0
        self.best_model = None

    def train(self):
        # Main training loop
        for epoch in range(self.cfg.model.num_epochs):
            self.model.train()
            running_loss = 0.0
            
            for batch in self.train_dataloader:
                inputs = batch[0].to(self.cfg.device)
                self.optimizer.zero_grad()
                all_outputs = self.model(inputs)
                
                # Separate the outputs based on whether the RBF layer and/or VAE are used

                if self.use_rbf:
                    outputs, _, rbf_out = all_outputs
                else:
                    outputs, _, _ = all_outputs
                    rbf_out = None
                                               
                # Compute the main loss
                loss = self.criterion(outputs, inputs)

                # 【毕设修改 M2】叠加 RBF 中心多样性 / 尺度正则（系数为 0 时不生效）
                if self.use_rbf and (self.center_div_lambda > 0 or self.gamma_reg_lambda > 0):
                    reg_loss, reg_details = self.model.rbf_layer.regularization(
                        center_div_lambda=self.center_div_lambda,
                        gamma_reg_lambda=self.gamma_reg_lambda,
                        gamma_reg_target=self.gamma_reg_target,
                    )
                    loss = loss + reg_loss
                    self.last_reg_details = reg_details

                # 【毕设修改 M5】密度对齐损失
                # 论文把"RBF 输出高"解释为"与正常数据相似"，但原训练目标里没有任何
                # 一项在推动这件事：RBF 层只靠重建误差回传的梯度被动调整，
                # 于是推理时 (1 - εs) 的判别能力不可控。
                # 这里把"正常训练窗口的平均 RBF 输出应贴近目标值"显式写进损失，
                # 让相似度分支有一个明确、可解释的优化目标。
                # 注意它与 M1 是配套的：只有当 RBF 不再顶替主干特征（M1）时，
                # 相似度分支才可以被单独塑形而不损害重建。
                if self.use_rbf and self.density_align_lambda > 0 and rbf_out is not None:
                    z_mean = rbf_out.mean(dim=(1, 2))
                    density_loss = ((z_mean - self.density_align_target) ** 2).mean()
                    loss = loss + self.density_align_lambda * density_loss
                    self.last_reg_details["density"] = float(density_loss.detach())
                
                # Backpropagation and optimization
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.model.clip_grad) 

                self.optimizer.step()
                running_loss += loss.item()

            # Compute the average loss for this epoch
            epoch_loss = running_loss / len(self.train_dataloader)

            # Evaluate the model on the test set
            test_loss = self.test_model()
            
            # Step the scheduler
            self.scheduler.step(test_loss)
                        
            # Print the loss every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch [{epoch + 1}/{self.cfg.model.num_epochs}], Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")

            # Check for early stopping
            if test_loss < self.best_loss:
                self.best_loss = test_loss
                self.counter = 0
                # Save the best model
                torch.save(self.model.state_dict(), self.save_path)

            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    # Load the best model before returning
                    self.model.load_state_dict(torch.load(self.save_path))
                    break
        
        # Load the best model after training
        self.model.load_state_dict(torch.load(self.save_path))

        # Delete the model file
        os.remove(self.save_path)
        
        return self.model

        

    def test_model(self):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for inputs in self.test_dataloader:
                inputs = inputs[0].to(self.cfg.device)
                outputs, _, _ = self.model(inputs)
                    
                loss = self.criterion(outputs, inputs)
                running_loss += loss.item()

        return running_loss / len(self.test_dataloader)



