import torch
import torchvision
from torch.utils.data import DataLoader

    def cal_distance(self,f_out,f_rec,vit = 343):
        #f_out:此时的输出信号频率，f_rec:此时的输入信号频率
        delta_f = np.min([f_out-f_rec,self.haute_frequency-f_rec+f_out-self.bas_frequency])
        pente_f = (self.haute_frequency-self.bas_frequency)/self.swept_last
        return delta_f/pente_f*vit/2
    
    def cal_forfreq(self,i,f_out,distance,vit = 343, lap = 0.005):
        #计算在当前输出信号下，达到distance+-lap*distance所需的频率，并计算对应强度和
        pente_f = (self.haute_frequency-self.bas_frequency)/self.swept_last
        fmax = f_out-(distance-lap)*pente_f*2/vit
        fmin = f_out-(distance+lap)*pente_f*2/vit
        pas = self.axe_freq
        if fmax>self.bas_frequency and fmin>self.bas_frequency:
            n_fmax = fmax//self.axe_freq_pas
            n_fmin = fmin//self.axe_freq_pas
            n_fmax = n_fmax.astype(int)
            n_fmin = n_fmin.astype(int)
            line = self.tftable[n_fmin:n_fmax,i]
            return np.sum(line)
        elif fmax>self.bas_frequency and fmin<self.bas_frequency:
            fmin = self.haute_frequency-self.bas_frequency+fmin
            n_fmax = fmax//self.axe_freq_pas
            n_fmin = fmin//self.axe_freq_pas
            n_bas = self.bas_frequency//self.axe_freq_pas
            n_haute = self.haute_frequency//self.axe_freq_pas
            n_fmax = n_fmax.astype(int)
            n_fmin = n_fmin.astype(int)
            n_haute = n_haute.astype(int)
            n_bas = n_bas.astype(int)
            line1 = self.tftable[n_bas:n_fmax,i]
            line2 = self.tftable[n_fmin:n_haute,i]
            return np.sum(line1)+np.sum(line2)
        elif fmax<self.bas_frequency and fmin<self.bas_frequency:
            fmin = self.haute_frequency-self.bas_frequency+fmin
            fmax = self.haute_frequency-self.bas_frequency+fmax
            n_fmax = fmax//self.axe_freq_pas
            n_fmin = fmin//self.axe_freq_pas
            n_fmax = n_fmax.astype(int)
            n_fmin = n_fmin.astype(int)
            line = self.tftable[n_fmin:n_fmax,i]
            return np.sum(line)
        elif fmax>self.haute_frequency and fmin<self.haute_frequency:
            fmax = self.bas_frequency-(self.haute_frequency-fmax)
            n_fmax = fmax//self.axe_freq_pas
            n_fmin = fmin//self.axe_freq_pas
            n_bas = self.bas_frequency//self.axe_freq_pas
            n_haute = self.haute_frequency//self.axe_freq_pas
            n_fmax = n_fmax.astype(int)
            n_fmin = n_fmin.astype(int)
            n_haute = n_haute.astype(int)
            n_bas = n_bas.astype(int)
            line1 = self.tftable[n_bas:n_fmax,i]
            line2 = self.tftable[n_fmin:n_haute,i]
            return np.sum(line1)+np.sum(line2)
        elif fmax>self.haute_frequency and fmin>self.haute_frequency:
            fmin = self.bas_frequency-(self.haute_frequency-fmin)
            fmax = self.bas_frequency-(self.haute_frequency-fmax)
            n_fmax = fmax//self.axe_freq_pas
            n_fmin = fmin//self.axe_freq_pas
            n_fmax = n_fmax.astype(int)
            n_fmin = n_fmin.astype(int)
            line = self.tftable[n_fmin:n_fmax,i]
            return np.sum(line)
        else:
            return 0
       
        
    def record_gene(self):
        self.general_sweptonde()
        input('信号生成完毕，输入任何键开始测试')
        self.pandr()
        print('测试完毕，请等待结果分析')
        self.get_data('record.wav')
        self.get_refer_data()
        self.make_tf()
        self.make_td_d2f()

    def record_nogene(self):
        input('现在开始测试：')
        self.pandr()
        print('测试结束，等待结果')
        self.get_data('record.wav')
        self.make_tf()
        self.make_td_d2f()