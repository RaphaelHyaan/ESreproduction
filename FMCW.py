import wave
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import time
import struct
import scipy.signal as signal

class FMCW():


    def __init__(self):

        self.c = 343                                #声速
        #   信号生成
        ##  采样
        self.sample_rate = 44100.0                   #每秒采样次数
        self.amplitude = 16000                       #振幅，以32767为上限

        ##  扫频
        self.swept_last = 0.012                                     #每次扫频持续时间
        self.chirp_last = int(self.swept_last*self.sample_rate)     #每次扫频采样次数

        self.chirp_nums = 200                          #播放的啁啾数
        self.chirp_l = np.array([15000,17000])          #左声道啁啾频率区间
        self.chirp_r = np.array([12000,14000])          #右声道啁啾频率区间

        #   信号读写
        self.chunk = self.chirp_last                 #缓冲区帧数
        self.format = pyaudio.paInt16                #采样位数
        ##  写

        ### 保存路径
        self.path_in = "chirp_lr.wav"                    #测试音频        
        self.path_out = "record.wav"                   #录音音频
        ##  读
        self.doc_frame_rate = 0                      #从文件获得帧率
        self.doc_frame_nums =  0                     #从文件获得帧数
        self.nchannels = 0                           #从文件获取的声道数

        self.wave_t = np.ones(0)                    #输出声波
        self.wave_r = np.ones(0)                    #接收声波
        self.wave_t_f = np.ones(0)                   #滤波后输出声波
        self.wave_r_f = np.ones(0)               #滤波后接收声波


    #信号录制
    def pandr(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(2),
                        channels=2,
                        rate=int(self.sample_rate),
                        input=True,
                        output=True,
                        frames_per_buffer=self.chunk)
        wf_i = wave.open(self.path_in, 'rb')  # 读 wav 文件
        wf = wave.open(self.path_out, 'wb')  # 打开 wav 文件。
        wf.setnchannels(2)  # 声道设置
        wf.setsampwidth(p.get_sample_size(self.format))  # 采样位数设置
        wf.setframerate(self.sample_rate)  # 采样频率设置
 
        data = wf_i.readframes(self.chirp_last)  # 读数据
        for i in range(0, self.chirp_nums):
            stream.write(data)
            datao = stream.read(self.chirp_last,exception_on_overflow = False)
            wf.writeframes(datao)  # 写入数据     
        stream.close()
        p.terminate()
        wf.close()

        pass

    #获得波
    def get_data(self,wavfile = 'record.wav'):
        #获取数据
        wf=wave.open(wavfile,"rb")              #打开wav
        p = pyaudio.PyAudio()                   #创建PyAudio对象
        params = wf.getparams()                 #参数获取
        self.nchannels, sampwidth, self.doc_frame_rate, self.doc_frame_nums = params[:4]
        stream = p.open(format=p.get_format_from_width(sampwidth),
                        channels=self.nchannels, 
                        rate=self.doc_frame_rate,
                        output=True)            #创建输出流
                                                #读取完整的帧数据到str_data中，这是一个string类型的数据
        str_data = wf.readframes(self.doc_frame_nums)
        wf.close()                              #关闭wave    
        self.wave_r = np.frombuffer(str_data, dtype=np.int16)
        self.wave_r = np.reshape(self.wave_r,[self.doc_frame_nums,self.nchannels]).astype(np.int64)
        '''
        plt.figure()
        plt.subplot(2,1,1)
        plt.specgram(self.wave_r[300*529:305*529,1],Fs = self.doc_frame_rate)
        plt.subplot(2,1,2)
        plt.specgram(self.wave_r[300*529:305*529,0],Fs = self.doc_frame_rate)
        plt.show()'''

        return self.wave_r

    #获得参考波
    def get_refer_data(self):
        #获取数据
        wavfile = self.path_in
        wf=wave.open(wavfile,"rb")              #打开wav
        p = pyaudio.PyAudio()                   #创建PyAudio对象
        params = wf.getparams()                 #参数获取
        nchannels, sampwidth, doc_frame_rate, self.doc_frame_nums = params[:4]
        stream = p.open(format=p.get_format_from_width(sampwidth),
                        channels=nchannels, 
                        rate=self.doc_frame_rate,
                        output=True)            #创建输出流
                                                #读取完整的帧数据到str_data中，这是一个string类型的数据
        str_data = wf.readframes(self.doc_frame_nums)
        self.wave_t= np.frombuffer(str_data, dtype=np.int16)
        self.wave_t = np.reshape(self.wave_t,[self.doc_frame_nums,nchannels]).astype(np.int64)
        '''        
        plt.figure()
        plt.subplot(2,1,1)
        plt.specgram(self.wave_t[:,1],Fs = doc_frame_rate)
        plt.subplot(2,1,2)
        plt.specgram(self.wave_t[:,0],Fs = doc_frame_rate)
        plt.show()'''
        wf.close()                              #关闭wave   

    #滤波    
    def filtre_bb(self):
        fl = self.chirp_l*2/self.doc_frame_rate
        bl,al = signal.butter(8,fl,'bandpass')
        fr = self.chirp_r*2/self.doc_frame_rate
        br,ar = signal.butter(8,fr,'bandpass')
        self.wave_t_f = np.zeros(np.shape(self.wave_t))
        self.wave_t_f[:,0] = signal.filtfilt(bl,al,self.wave_t[:,0])
        self.wave_t_f[:,1] = signal.filtfilt(br,ar,self.wave_t[:,1])
        size = list(np.shape(self.wave_r))
        size[1]*=2
        self.wave_r_f = np.zeros(size)
        self.wave_r_f[:,0] = signal.filtfilt(bl,al,self.wave_r[:,0])
        self.wave_r_f[:,2] = signal.filtfilt(br,ar,self.wave_r[:,0])
        self.wave_r_f[:,1] = signal.filtfilt(bl,al,self.wave_r[:,1])
        self.wave_r_f[:,3] = signal.filtfilt(br,ar,self.wave_r[:,1])

        '''
        plt.figure()
        plt.subplot(3,2,1)
        plt.specgram(self.wave_t_f[:,0],Fs = self.doc_frame_rate)
        plt.subplot(3,2,2)
        plt.specgram(self.wave_t_f[:,1],Fs = self.doc_frame_rate)
        plt.subplot(3,2,3)
        plt.specgram(self.wave_r_f[:,0],Fs = self.doc_frame_rate)
        plt.subplot(3,2,4)
        plt.specgram(self.wave_r_f[:,1],Fs = self.doc_frame_rate)
        plt.subplot(3,2,5)
        plt.specgram(self.wave_r_f[:,2],Fs = self.doc_frame_rate)
        plt.subplot(3,2,6)
        plt.specgram(self.wave_r_f[:,3],Fs = self.doc_frame_rate)
        plt.show()'''

        return 0

    #使用矩阵方法计算cc
    def cc_matrice(self,foi):
        #尝试使用矩阵方式获得lag
        N = self.chirp_last
        vtx = self.wave_t_f[0:N,:]
        vtx = np.repeat(vtx,2,axis = 1)
        vrx = self.wave_r_f[foi*N:(foi+1)*N,:]
        r2 = np.zeros((2*N-1,4))
        for i in range(4):
            r2[:,i] = np.correlate(vtx[:,i], vrx[:,i], mode = 'full')
        lag2 = np.argmax(r2,axis = 0)
        return r2,lag2

    #计算距离
    def distance_matrix(self):
        N =self.chirp_last
        m_r = np.zeros((2*N-1,self.chirp_nums,4))
        m_d = np.zeros((2*N-1,self.chirp_nums,4))
        l_lap = np.zeros((self.chirp_nums,4))
        for i in range(self.chirp_nums):
            m_r[:,i,:],l_lap[i,:] = self.cc_matrice(i)
        m_d = m_r*self.c/(2*self.sample_rate)
        
        t_axe = np.linspace(0,self.chirp_nums*N/self.sample_rate,self.chirp_nums-1)
        c_axe = np.linspace(-N+1,N,2*N-1)
        d_axe = c_axe*self.c/(2*self.sample_rate)

        m_d_d = np.diff(m_d,axis = 1)

        '''        
        plt.figure()
        plt.subplot(1,1,1)
        plt.plot(t_axe,l_lap,label = ['1','2','3','4'])
        plt.title('Lap_list')
        plt.legend()
        plt.show()'''

        self.print_table(t_axe,d_axe,m_d_d,12,6)

        return 0

    def print_table(self,t_axe,d_axe,table,vmax = 8,vmin = 7):
        #输出一个三维的表格
        plt.figure()
        plt.subplot(2,2,1)
        plt.pcolormesh(t_axe,d_axe,np.log(np.abs(table[:,:,0])),vmax = vmax,vmin = vmin)
        plt.title('0:left_hf')
        plt.subplot(2,2,2)
        plt.pcolormesh(t_axe,d_axe,np.log(np.abs(table[:,:,1])),vmax = vmax,vmin = vmin)
        plt.title('1:right_hf')
        plt.subplot(2,2,3)
        plt.pcolormesh(t_axe,d_axe,np.log(np.abs(table[:,:,2])),vmax = vmax,vmin = vmin)
        plt.title('2:left_bf')
        plt.subplot(2,2,4)
        plt.pcolormesh(t_axe,d_axe,np.log(np.abs(table[:,:,3])),vmax = vmax,vmin = vmin)
        plt.title('1:right_bf')
        plt.show()

    def print_list(t_axe,list,title = ['左侧高频信号','右侧高频信号','左侧低频信号','右侧低频信号'],label = [1,2,3,4]):
        plt.figure()
        plt.rcParams['font.sans-serif'] = ['fangsong']
        plt.rcParams['axes.unicode_minus']=False
        plt.subplot(2,2,1)
        plt.plot(t_axe,list[:,0],label = label[0])
        plt.title(title[0])
        plt.subplot(2,2,2)
        plt.plot(t_axe,list[:,1],label = label[1])
        plt.title(title[1])
        plt.subplot(2,2,3)
        plt.plot(t_axe,list[:,2],label = label[2])
        plt.title(title[2])
        plt.subplot(2,2,4)
        plt.plot(t_axe,list[:,3],label = label[3])
        plt.title(title[3])
        plt.legend()
        plt.show()

f = FMCW()

#f.record_gene()
#f.pandr()

f.get_data()
f.get_refer_data()
f.filtre_bb()

begin = time.time()
f.distance_matrix()
p1 = time.time()

begin = time.time()
r,lap = f.cc_matrice(f.unitaire(143))
end1 = time.time()


end2 = time.time()
print('新方法时间：%f，老方法时间：%f' %(p1-begin,4*(end2-p1)))


input()

