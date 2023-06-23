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
    
    def c_fmcw_list(self):
        N = self.chirp_last
        c_table = np.zeros((2*N-2,self.chirp_nums,4))
        distance_table = np.zeros((2*N-2,self.chirp_nums,4))
        lap_list = np.zeros((self.chirp_nums,4))

        t_axe = np.linspace(0,self.chirp_nums*N/self.sample_rate,self.chirp_nums)
        c_axe = np.linspace(-N+1,N-1,2*N-2)
        d_axe = c_axe*self.c/(2*self.sample_rate)

        vtx = np.abs(self.wave_t_f)
        vrx = np.abs(self.wave_r_f)
        for i in range(self.chirp_nums):
            lap_list[i,0],c_table[:,i,0] = self.cross_correction_list(vtx[:,1],vrx[:,0],i)
            lap_list[i,1],c_table[:,i,1] = self.cross_correction_list(vtx[:,1],vrx[:,2],i)
            lap_list[i,2],c_table[:,i,2] = self.cross_correction_list(vtx[:,0],vrx[:,1],i)
            lap_list[i,3],c_table[:,i,3] = self.cross_correction_list(vtx[:,0],vrx[:,3],i)
        
        dc_table = np.zeros((2*N-2,self.chirp_nums,4))
        for i in range(self.chirp_nums):
            dc_table[:,i,:] = c_table[:,i,:]-c_table[:,i-1,:]
        
        distance_table = dc_table*self.c/(2*self.sample_rate)
        '''
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(t_axe,lap_list,label = ['1','2','3','4'])
        plt.title('Lap_list')
        plt.legend()
        plt.subplot(3,1,2)
        plt.pcolormesh(t_axe,c_axe,np.log(np.abs(c_table[:,:,3])))
        plt.title('C_table')
        plt.subplot(3,1,3)
        plt.pcolormesh(t_axe,d_axe,np.log(np.abs(distance_table[:,:,1])))
        plt.title('Distance_table')
        plt.show()'''


        return lap_list,c_table,distance_table

    def cross_correction_list(self,vtx,vrx,foi):
        """
        vtx: 相应输出信号数组
        vrx: 相应接收信号数组
        foi: 正在处理的啁啾个数,比如在比较第2个啁啾,foi = 2
        """
        N = self.chirp_last
        r_list = np.zeros(2*N-2)
        for n in range(-N+1,N-1):
            r_list[n+N-1] = self.cross_correction(vtx,vrx,N,n,foi)
        Lag = np.argmax(r_list)-N+1
        return Lag, r_list

    def cross_correction(self,vtx,vrx,N,n,foi):
        if n>=0:
            sum = 0
            for m in range(0,N-n-1):
                dsum = vtx[m]*vrx[int(foi*N)+m+n]
                sum += dsum
            return sum/np.max([(N-n),100])
        else:
            sum = 0
            for m in range(0,N+n-2):
                sum += vrx[int(foi*N)+m]*vtx[m-n]
            return sum/np.max([(N-n),100])       

    def cc_matrice(self,foi):
        #尝试使用矩阵方式获得lag
        N = self.chirp_last
        vtx = self.wave_t_f[0:N,:]
        vtx = np.repeat(vtx,2,axis = 1)
        vrx = self.wave_r_f[foi*N:(foi+1)*N,:]

        mat = np.einsum('ik,jk->ijk',vtx,vrx)   #计算每个batch下两个向量的外积


        
        r = np.zeros((2*N-1,4))


        for n in range(-N+1,N):
            r[n+N-1,:] = np.trace(mat,n,axis1 = 0,axis2 = 1)/np.max([N-np.abs(n),200])


        lag = np.zeros(4)
        lag = np.argmax(r,axis = 0)
        return r,lag

    def distance_matrix(self):
        N =self.chirp_last
        m_r = np.zeros((2*N-1,self.chirp_nums,4))
        m_d = np.zeros((2*N-1,self.chirp_nums,4))
        l_lap = np.zeros((self.chirp_nums,4))
        for i in range(self.chirp_nums):
            m_r[:,i,:],l_lap[i,:] = self.cc_matrice(i)

        m_d = m_r*self.c/(2*self.sample_rate)
        
        t_axe = np.linspace(0,self.chirp_nums*N/self.sample_rate,self.chirp_nums)
        c_axe = np.linspace(-N+1,N,2*N-1)
        d_axe = c_axe*self.c/(2*self.sample_rate)

        m_d_d = np.zeros((2*N-1,self.chirp_nums,4))
        for i in range(self.chirp_nums-1):
            m_d_d[:,i,:] = m_d[:,i+1,:]-m_d[:,i,:]

        '''        
        plt.figure()
        plt.subplot(1,1,1)
        plt.plot(t_axe,l_lap,label = ['1','2','3','4'])
        plt.title('Lap_list')
        plt.legend()
        plt.show()'''

        #self.print_table(t_axe,d_axe,m_d_d,4,2)

        return 0

    def print_table(self,t_axe,d_axe,table,vmax = 8,vmin = 7):
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

    def unitaire(self,list):
        #标准化
        mean = np.mean(np.abs(list))
        #return np.abs(list)
        return list

f = FMCW()

#f.record_gene()
f.pandr()

f.get_data()
f.get_refer_data()
f.filtre_bb()
begin = time.time()
f.distance_matrix()
p1 = time.time()

f.c_fmcw_list()
p2 = time.time()
'''
vtx = f.get_data('chirp_0.012.wav')
vrx = f.get_data('chirp_0.012rr.wav')
lap,list = f.cross_correction_list(vtx[:,0],vrx[:,0],0)
'''
'''begin = time.time()
r,lap = f.cc_matrice(f.unitaire(81))
end1 = time.time()'''

'''lap,list = f.cross_correction_list(f.wave_t_f[:,0],f.wave_r_f[:,0],81)
end2 = time.time()
plt.figure()
plt.rcParams['font.sans-serif'] = ['fangsong']
plt.rcParams['axes.unicode_minus']=False
t = np.linspace(-f.chirp_last+1,f.chirp_last,2*f.chirp_last-1)
plt.subplot(2,2,1)
plt.plot(t,r[:,0])
plt.title('左侧高频信号')
plt.subplot(2,2,2)
plt.plot(t,r[:,1],label = [1])
plt.title('右侧高频信号')
plt.subplot(2,2,3)
plt.plot(t,r[:,2],label = [2])
plt.title('左侧低频信号')
plt.subplot(2,2,4)
plt.plot(t,r[:,3],label = [3])
plt.title('右侧低频信号')
#plt.plot(t,list,label ='原')
plt.legend()
plt.show()
print('新方法时间：%f，老方法时间：%f' %(end1-begin,4*(end2-end1)))
'''
print('新方法时间：%f，老方法时间：%f' %(p1-begin,p2-p1))
input()

