import wave
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import os

class C_FMCW():
    def __init__(self,pathout = 'record.wav',last = 2):

        self.c = 343                                 #声速
        #   信号生成
        ##  采样
        self.sample_rate = 44100.0                   #每秒采样次数

        ## 扫频
        self.swept_last = 0.012                                     #每次扫频持续时间
        self.chirp_last = int(self.swept_last*self.sample_rate)     #每次扫频采样次数

        ## 播放
        self.chirp_nums = int(last/self.swept_last)             #播放的啁啾数
        self.remove_nums = 50                                   #休息时间
        self.chirp_l = np.array([17000,18000])                  #左声道啁啾频率区间
        self.chirp_r = np.array([15000,16000])                  #右声道啁啾频率区间

        ## 缓冲区读写
        self.chunk = self.chirp_last                 #缓冲区帧数
        self.format = pyaudio.paInt16                #采样位数


        ## 路径
        self.path_in = "chirp/chirp_lr_17000_18000_15000_16000.wav"                    #测试音频   
        self.name = pathout
        self.save_name = ''
        self.path_out = 'data/wav/'+pathout+'/'+pathout+'.wav'                       #录音音频

        ## 读
        self.doc_frame_rate = self.sample_rate       #从文件获得帧率
        self.doc_frame_nums =  0                     #从文件获得帧数
        self.nchannels = 0                           #从文件获取的声道数

        ## 目标波
        self.wave_t = np.ones(0)                    #输出声波
        self.wave_r = np.ones(0)                    #接收声波
        self.wave_t_f = np.ones(0)                  #滤波后输出声波
        self.wave_r_f = np.ones(0)                  #滤波后接收声波

        #滤波器文件
        fl = self.chirp_l*2/self.doc_frame_rate
        self.bl,self.al = signal.butter(8,fl,'bandpass')
        fr = self.chirp_r*2/self.doc_frame_rate
        self.br,self.ar = signal.butter(8,fr,'bandpass')

    
    def pandr(self):
        ''' 基础的信号录制功能 '''
        p = pyaudio.PyAudio()

        stream = p.open(format=p.get_format_from_width(2),
                        channels=2,
                        rate=int(self.sample_rate),
                        input=True,
                        output=True,
                        frames_per_buffer=self.chunk)
        
        wf_i = wave.open(self.path_in, 'rb')  # 读 wav 文件
        wf = wave.open(self.path_out, 'wb')  # 打开 wav 文件
        wf.setnchannels(2)  # 声道设置
        wf.setsampwidth(p.get_sample_size(self.format))  # 采样位数设置
        wf.setframerate(self.sample_rate)  # 采样频率设置
 
        data = wf_i.readframes(self.chirp_last)  # 读数据
        for i in range(0, self.chirp_nums):
            if i == self.remove_nums:
                print('*')
            stream.write(data)
            datao = stream.read(self.chirp_last,exception_on_overflow = False)
            #a = np.frombuffer(datao)在未来可尝试的改进方案

            wf.writeframes(datao)  # 写入数据     
        stream.close()
        p.terminate()
        wf.close()

    def get_data(self,T = None):
        '''从文件读取接收数组
        T:是否转置'''
        
        wavfile = self.path_out
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

        #滤波
        size = list(np.shape(self.wave_r))
        size[1]*=2
        self.wave_r_f = np.zeros(size)
        self.wave_r_f[:,0] = signal.filtfilt(self.bl,self.al,self.wave_r[:,0])
        self.wave_r_f[:,2] = signal.filtfilt(self.br,self.ar,self.wave_r[:,0])
        self.wave_r_f[:,1] = signal.filtfilt(self.bl,self.al,self.wave_r[:,1])
        self.wave_r_f[:,3] = signal.filtfilt(self.br,self.ar,self.wave_r[:,1])

        if T is not None:
            self.wave_r_f = self.wave_r_f.T

        return self.wave_r

    def get_refer_data(self,T = None):
        '''从文件获取发射信号数组,T:是否转置'''
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

        #滤波
        self.wave_t_f = np.zeros(np.shape(self.wave_t))
        #self.wave_t_f[:,0] = signal.filtfilt(self.bl,self.al,self.wave_t[:,0])
        #self.wave_t_f[:,1] = signal.filtfilt(self.br,self.ar,self.wave_t[:,1])
        self.wave_t_f = self.wave_t
        if T is not None:
            self.wave_t_f = self.wave_t_f.T
        wf.close()                              #关闭wave   

    def load_refer_data(self,filename = r'npy\referwave.npy'):
        '''从numpy数组文件读取发射信号数组'''
        self.wave_t_f = self.load(filename)

    def print_wave_f(self):
        '''绘制FMCW信号specgram波形'''
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
        plt.show()
        return

    def c_corre(self,foi):
        '''计算互相关'''
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

    def distance_matrix(self,show = False,save = False):
        '''获得距离矩阵
        m_d 未差分的距离矩阵
        m_d_d 差分后的距离矩阵
        m_d_d_2 对其后的距离矩阵
        '''
        N =self.chirp_last
        m_r = np.zeros((2*N-1,self.chirp_nums-self.remove_nums,4))
        m_d = np.zeros((2*N-1,self.chirp_nums-self.remove_nums,4))
        #m_d_d = np.zeros((2*N-1,self.chirp_nums-2,4))
        m_d_d_2 = np.zeros((201,self.chirp_nums-self.remove_nums-1,4))

        l_lap = np.zeros((self.chirp_nums-self.remove_nums,4))
        for i in range(self.chirp_nums-self.remove_nums):
            m_r[:,i,:],l_lap[i,:] = self.c_corre(i+self.remove_nums)
        m_d = m_r*self.c/(2*self.sample_rate)
        
        t_axe = np.linspace(0,(self.chirp_nums-self.remove_nums)*N/self.sample_rate,self.chirp_nums-self.remove_nums-1)
        c_axe = np.linspace(-100,100,201)
        cn_axe = np.linspace(-N+1,N,2*N-1)
        d_axe = c_axe*self.c/(2*self.sample_rate)
        dn_axe= cn_axe*self.c/(2*self.sample_rate)



        m_d_d = np.diff(m_d,axis = 1)
        for i in range(0,4):
    
            lap_i = np.argmax(m_d_d[:,:,i],axis = 0)
            if np.mean(lap_i[lap_i>N])-np.mean(lap_i[lap_i<N]) <50:
                lap = np.mean(lap_i).astype(int)
            elif np.shape(lap_i[lap_i>N])>np.shape(lap_i[lap_i<N]):
                lap = np.mean(lap_i[lap_i>N]).astype(int)
            else:
                lap = np.mean(lap_i[lap_i<N]).astype(int)
            
            m_d_d_2[:,:,i] = m_d_d[lap-100:lap+101,:,i]
        if show or save:
            self.print_table(t_axe,d_axe,m_d_d_2[:,:],13,10,save = save,show = show,mod_max = -0.8,mod_min = 0,cmap = 'gray')
        return m_d_d

    def print_table(self,t_axe,d_axe,table,vmax = 12,vmin = 10,mod_max = -0.8,mod_min = 0,save = False,show = True,cmap = 'jet'):
        '''输出一个三维的表格，是否保存、是否输出
        mod:对高频信号和低频信号强度不同的修正'''
        #plt.figure()
        plt.subplot(2,2,1)
        plt.pcolormesh(t_axe,d_axe,np.log(np.abs(table[:,:,0])),vmax = vmax,vmin = vmin,cmap='jet',norm="log",shading =  'gouraud')
        plt.title('0:left_hf')
        plt.subplot(2,2,2)
        plt.pcolormesh(t_axe,d_axe,np.log(np.abs(table[:,:,1])),vmax = vmax,vmin = vmin,cmap='jet',norm="log",shading =  'gouraud')
        plt.title('1:right_hf')
        plt.subplot(2,2,3)
        plt.pcolormesh(t_axe,d_axe,np.log(np.abs(table[:,:,2])),vmax = vmax+mod_max,vmin = vmin+mod_min,cmap='jet',norm="log",shading =  'gouraud')
        plt.title('2:left_bf')
        plt.subplot(2,2,4)
        plt.pcolormesh(t_axe,d_axe,np.log(np.abs(table[:,:,3])),vmax = vmax+mod_max,vmin = vmin+mod_min,cmap='jet',norm="log",shading =  'gouraud')
        plt.title('1:right_bf')
        if show:
            plt.show()
        if save:
            for i in range(0,4):
                if i >=2:
                    vmax += mod_max
                    vmin += mod_min
                ax = plt.figure()
                plt.pcolormesh(t_axe,d_axe,np.log(np.abs(table[:,:,i])),vmax = vmax,vmin = vmin,cmap=cmap,norm="log",shading =  'gouraud')
                plt.axis('off')   # 去坐标轴
                plt.xticks([])    # 去 x 轴刻度
                plt.yticks([])    # 去 y 轴刻度
                if save:
                    if not os.path.exists('data/image/'+self.save_name):
                        os.mkdir('data/image/'+self.save_name)
                    plt.savefig('data/image/'+self.save_name+'/image_0'+str(i)+'.jpg', dpi=300, bbox_inches='tight', pad_inches = -0.1)
                if show:
                    plt.show()
                plt.close()

    def save(self,a,filename):
        np.save(filename,a)
    
    def load(self,filename):
        return np.load(filename)
    
    def mkdir(self,types = ['wav','npy','image']):
        for type in types:
            path = 'data/'+type+'/'+self.name
            pe = os.path.exists(path)
            if not pe:
                os.makedirs(path)
        self.name = self.name+'/'+self.name
        return 0