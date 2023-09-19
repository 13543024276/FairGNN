import torch.nn as nn
from models.GCN import GCN,GCN_Body
from models.GAT import GAT,GAT_body
import torch

def get_model(nfeat, args):
    if args.model == "GCN":
        model = GCN_Body(nfeat,args.num_hidden,args.dropout)
    elif args.model == "GAT":
        heads =  ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        model = GAT_body(args.num_layers,nfeat,args.num_hidden,heads,args.dropout,args.attn_drop,args.negative_slope,args.residual)
    else:
        print("Model not implement")
        return

    return model

class FairGNN(nn.Module):

    def __init__(self, nfeat, args):
        super(FairGNN,self).__init__()

        nhid = args.num_hidden
        dropout = args.dropout
        self.estimator = GCN(nfeat,args.hidden,1,dropout)
        self.GNN = get_model(nfeat,args)
        self.classifier = nn.Linear(nhid,1)
        self.adv = nn.Linear(nhid,1)

        G_params = list(self.GNN.parameters()) + list(self.classifier.parameters()) + list(self.estimator.parameters())
        self.optimizer_G = torch.optim.Adam(G_params, lr = args.lr, weight_decay = args.weight_decay)
        self.optimizer_A = torch.optim.Adam(self.adv.parameters(), lr = args.lr, weight_decay = args.weight_decay)

        self.args = args
        self.criterion = nn.BCEWithLogitsLoss()

        self.G_loss = 0
        self.A_loss = 0

    def forward(self,g,x):
        s = self.estimator(g,x)
        z = self.GNN(g,x)
        y = self.classifier(z)
        return y,s
    
    def optimize(self,g,x,labels,idx_train,sens,idx_sens_train):
        self.train()

        ## update Adv
        self.GNN.requires_grad_(False)
        self.classifier.requires_grad_(False)
        self.estimator.requires_grad_(False)
        self.adv.requires_grad_(True)
        s = self.estimator(g, x)
        # s_score:是敏感属性估计器预测出来的概率值：不参与梯度计算，并且训练部分的概率采取真实值，无参与计算图的构建
        s_score = torch.sigmoid(s.detach())
        # s_score = (s_score > 0.5).float()
        s_score[idx_sens_train] = sens[idx_sens_train].unsqueeze(1).float()
        for epoch in range(3):
            self.optimizer_A.zero_grad()

            h = self.GNN(g, x)
            s_g = self.adv(h.detach())

            self.A_loss = self.criterion(s_g, s_score)
            self.A_loss.backward()
            self.optimizer_A.step()

        ### update E, G
        # self.estimator.requires_grad_(False)
        # self.adv.requires_grad_(False)
        self.GNN.requires_grad_(True)
        self.classifier.requires_grad_(True)
        # self.estimator.requires_grad_(True)
        self.adv.requires_grad_(False)

        self.optimizer_G.zero_grad()
        #只是得到估计值，并不是概率
        # s = self.estimator(g,x)
        h = self.GNN(g,x)
        # 只是得到估计值，并不是概率
        y = self.classifier(h)
        #对抗器输出来的无sigmoid的数
        s_g = self.adv(h)
        #s_score:是敏感属性估计器预测出来的概率值：不参与梯度计算，并且训练部分的概率采取真实值，无参与计算图的构建
        # s_score = torch.sigmoid(s.detach())
        # # s_score = (s_score > 0.5).float()
        # s_score[idx_sens_train]=sens[idx_sens_train].unsqueeze(1).float()

        #计算出分类的预测值
        y_score = torch.sigmoid(y)

        #计算协方差损失
        self.cov =  torch.abs(torch.mean((s_score - torch.mean(s_score)) * (y_score - torch.mean(y_score))))
        #计算分类损失
        self.cls_loss = self.criterion(y[idx_train],labels[idx_train].unsqueeze(1).float())
        #计算对抗损失
        self.adv_loss = self.criterion(s_g,s_score)
        
        self.G_loss = self.cls_loss  + self.args.alpha * self.cov - self.args.beta * self.adv_loss
        self.G_loss.backward()

        # for name, parameters in self.estimator.named_parameters():
        #     print(name,parameters.grad)
        #     tensor1=(parameters.data.clone().detach())
        #     break
        self.optimizer_G.step()
        # for name, parameters in self.estimator.named_parameters():
        #     tensor2 = parameters.data.clone().detach()
        #     print(torch.allclose(tensor2,tensor1))
        #     break



