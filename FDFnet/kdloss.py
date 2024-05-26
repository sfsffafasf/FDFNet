import torch
import torch.nn as nn
import torch.nn.functional as F
class ATLoss(nn.Module):
    
    def __init__(self,student_channels,teacher_channels):
        super(ATLoss, self).__init__()
        # self.mse = nn.MSELoss(reduce= 'mean')
        self.SmoothL1Loss = nn.SmoothL1Loss()
        # self.layer_ful2 = nn.Sequential(nn.Conv2d(in_dim // 4, in_dim//8, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(in_dim//8),nn.LeakyReLU())
        # self.DiceLoss = DiceLoss()
        self.teacherconvs = nn.Sequential(nn.Conv2d(teacher_channels, student_channels, kernel_size=3, stride=1, padding=1),nn.MaxPool2d(kernel_size=2, stride=2), nn.BatchNorm2d(student_channels ))
        self.studentconvs = nn.Sequential(nn.Conv2d(student_channels, student_channels,kernel_size=3, stride=1, padding=1), nn.MaxPool2d(kernel_size=2, stride=2),nn.BatchNorm2d(student_channels ))
        # self.Similarity = Similarity()


    def forward(self,  x_student_dep,x_student_img,x_teacher_dep,x_teacher_img):
        # print(x_teacher_dep.shape,x_student_dep.shape)
        # x_teacher = self.teacherconvs(x_teacher_img+x_teacher_dep)
        # x_student = self.studentconvs(x_student_dep+x_student_img)
        x_teacher = self.teacherconvs(x_teacher_img+x_teacher_dep)
        x_student = self.studentconvs(x_student_dep+x_student_img)
        # print(x_student.shape, x_teacher.shape)#修改
        loss_PD =self.SmoothL1Loss(torch.sigmoid(x_student), torch.sigmoid(x_teacher))
        # print(loss_PD.type)
        loss = loss_PD
        return loss

class SCCL(torch.nn.Module):
    def __init__(self):
        super(SCCL, self).__init__()
        self.tau: float = 0.5
        student_channels = [32, 64, 160, 256]
        teacher_channels = 512
        # OUT_3s.float(), OUT_2s.float(), OUT_1s.float(), OUT_0s.float(), OUT_3.float()
        self.conv2 = nn.Conv2d(in_channels=160, out_channels=256, kernel_size=1,
                          stride=1)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1,
                          stride=1)
        self.conv0 = nn.Conv2d(in_channels=32, out_channels=256, kernel_size=1,stride=1)
        self.convt = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,z3: torch.Tensor):

        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z2))
        between_sim = f(self.sim(z1, z3))
        return -torch.log(
        between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        # return -torch.log(
        # between_sim.diag()
        #         # OUT_3s.float(), OUT_2s.float(), OUT_1s.float(), OUT_0s.float(), OUT_3.float()
    def loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, z4: torch.Tensor,z5: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        # teacherchannel = z2.shape[1]
        # studentchannel = z1.shape[1]
        # if teacherchannel != studentchannel:
        #     self.conv = nn.Conv2d(in_channels=teacherchannel, out_channels=studentchannel, kernel_size=1,
        #                           stride=1).cuda()
        z1 = F.interpolate(z1, size=(13, 13), mode='bilinear')
        z22 = F.interpolate(self.conv2(z2), size=(13, 13), mode='bilinear')
        z3 = F.interpolate(self.conv1(z3), size=(13, 13), mode='bilinear')
        z4 = F.interpolate(self.conv0(z4), size=(13, 13), mode='bilinear')
        z2 = self.convt(z5)

        z1 = z1+z22+z3+z4
        qz2 = 1-z2
        qz1 = 1-z1
        z1 = torch.sigmoid(z1)
        z2 = torch.sigmoid(F.interpolate(z2, size=(13, 13), mode='bilinear'))
        qz1 = torch.sigmoid(qz1)
        qz2 = torch.sigmoid(F.interpolate(qz2, size=(13, 13), mode='bilinear'))
        b, c, h, w = z1.shape
        h1 = z1.reshape(b, -1)/(h*w*b)
        h2 = z2.reshape(b, -1)/(h*w*b)

        qh1 = qz1.reshape(b, -1)/(h*w*b)
        qh2 = qz2.reshape(b, -1)/(h*w*b)

        l1 = self.semi_loss(h1, h2,qh2)/c
        l2 = self.semi_loss(qh1,qh2,h2)/c

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
class Similarity(nn.Module):
    ##Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author##
    def __init__(self):
        super(Similarity, self).__init__()

        # self.SmoothL1Loss = nn.SmoothL1Loss()

    def forward(self, g_s, g_t):
        # teacherchannel = g_t.shape[1]
        # studentchannel = g_s.shape[1]
        # if teacherchannel != studentchannel:
        #     self.conv = nn.Conv2d(in_channels=teacherchannel, out_channels=studentchannel, kernel_size=1,
        #                           stride=1).cuda()
        #     g_t = self.conv(g_t)
        return self.similarity_loss(g_s, g_t)

    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)#+self.SmoothL1Loss(f_s,f_t).mean(-1)
        return loss



class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Networkzij"""

    def __init__(self):
        super(DistillKL, self).__init__()
        self.T = 1
        self.pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.MultiscalePoolingt = MultiscalePooling()
        # self.DiceLoss = DiceLoss()
    def forward(self, ys,ys1,y):

        y_sM = self.pool_layer(ys+ys1)
        # print(y_sM.shape)
        y_t = self.pool_layer(y)
        # print(y_t.shape)
        p_s = F.log_softmax(y_sM/ self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        # print(p_s,p_t)
        # 蒸馏损失采用的是KL散度损失函数
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_t.shape[0]

        return loss
