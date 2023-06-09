import wave
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import time
import struct
import scipy.signal as signal

class FMCW():


    def __init__(self):
        #   信号生成
        ##  采样
        self.sample_rate = 44100.0                   #每秒采样次数
        self.amplitude = 16000                       #振幅，以32767为上限
        self.sample_last = 2                         #采样时间
        self.sample_nums = int(self.sample_last*self.sample_rate)                    #总采样次数
        ##  扫频
        self.bas_frequency = 12000                                  #最低频率
        self.haute_frequency = 16000                                #最高频率
        self.swept_last = 0.012                                     #每次扫频持续时间
        self.swept_nums = int(self.swept_last*self.sample_rate)     #每次扫频采样次数


        #   信号读写
        self.chunk = 1024                            #缓冲区帧数
        self.format = pyaudio.paInt16                #采样位数
        ##  写
        ### 实现不保存到本地
        self.outputsignal = bytes()                  #测试音频数据，未实现
        ### 保存路径
        self.path_in = "test.wav"                    #测试音频默认保存        
        self.path_out = "record.wav"                 #录音音频默认保存
        ##  读
        self.doc_frame_rate = 0                      #从文件获得帧率
        self.doc_frame_nums =  0                     #从文件获得帧数
        self.doc_wave = np.ones(0)                   #从文件获得的声音数组

        self.doc_refer_wave = np.ones(0)             #参考波
        

        #   信号处理
        self.axe_times = np.ones(0)             #时间轴
        self.axe_freq = np.ones(0)              #频率轴
        self.axe_distance = np.ones(0)          #距离轴
        self.refer_table = np.ones((0,0))       #参考表
        self.tftable = np.ones((0,0))           #时频表
        self.tdtable = np.ones((0,0))           #时间距离表
        self.dtdtable = np.ones((0,0))          #差分时间距离表
        self.axe_freq_pas = 0                   #频率变化步长


        #短时傅里叶变换
        self.nperseg = 1000
        self.noverlap= 200
        self.nfft = 5120

        self.doc_refer_diri = self.axe_times%0.12*(self.haute_frequency-self.bas_frequency)+self.bas_frequency

    # 信号生成和处理 
    def general_sweptonde(self,in_path = "test.wav"):
        self.path_in = in_path    
        #扫频信号生成
        #frequency = [self.bas_frequency+(self.haute_frequency-self.bas_frequency)*(x%self.swept_nums)/self.swept_nums for x in range(self.sample_nums)]
        #sine_wave = [np.cos(2 * np.pi * frequency[x] * x/self.sample_rate) for x in range(self.sample_nums)]
        #times = np.linspace(0,self.sample_nums/self.sample_rate,self.sample_nums)
        #sine_wave = signal.chirp(times,f0 = self.bas_frequency,t1 = self.swept_nums/self.sample_rate,f1 = self.haute_frequency,method = 'linear')
        times = np.linspace(0,self.swept_nums/self.sample_rate,self.swept_nums)
        sine_wave_1 = signal.chirp(times,f0 = self.bas_frequency,t1 = self.swept_nums/self.sample_rate,f1 = self.haute_frequency,method = 'linear')
        sine_wave = np.ones(0)
        for i in range(0,self.sample_nums,self.swept_nums):
            sine_wave = np.concatenate((sine_wave,sine_wave_1))

        nframes=self.sample_nums    #帧数
        comptype="NONE"             #是否压缩
        compname="not compressed"   #是否压缩
        nchannels=1                 #信道数
        sampwidth=2                 #样本宽度，一般为2字节

        wav_file=wave.open(in_path, 'w')
        wav_file.setparams((nchannels, sampwidth, int(self.sample_rate), nframes, comptype, compname))
        for s in sine_wave:
            wav_file.writeframes(struct.pack('h', int(s*self.amplitude)))    #h表示16位数
            #self.outputsignal+=struct.pack('h', int(s*self.amplitude))

    def play(self):
        #播放信号
        wf=wave.open(self.path_in,"rb")#打开wav

        p = pyaudio.PyAudio()
        stream = p.open(format=self.format,
                        channels=1,
                        rate=int(self.sample_rate),
                        output=True)
        '''
        data = self.outputsignal[0:self.chunk]
        base = self.chunk
        while base < self.sample_nums:
            stream.write(data)
            data = self.outputsignal[base:base+self.chunk]
        '''
        data = wf.readframes(self.chunk)  # 读数据
        while len(data) > 0:
            stream.write(data)
            data = wf.readframes(self.chunk)
        stream.stop_stream()
        stream.close()
        p.terminate

    def record(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format,
                        channels=1,
                        rate=int(self.sample_rate),
                        input=True,
                        frames_per_buffer=self.chunk)  # 打开流，传入响应参数
        wf = wave.open(self.path_out, 'wb')  # 打开 wav 文件。
        wf.setnchannels(1)  # 声道设置
        wf.setsampwidth(p.get_sample_size(self.format))  # 采样位数设置
        wf.setframerate(self.sample_rate)  # 采样频率设置

        for _ in range(0, int(self.sample_nums / self.chunk)):
            data = stream.read(self.chunk)
            wf.writeframes(data)  # 写入数据
        stream.stop_stream()  # 关闭流
        stream.close()
        p.terminate()
        wf.close()        
        pass
    
    def pandr(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(2),
                        channels=1,
                        rate=int(self.sample_rate),
                        input=True,
                        output=True,
                        frames_per_buffer=self.chunk)
        wf_i = wave.open(self.path_in, 'rb')  # 读 wav 文件
        wf = wave.open(self.path_out, 'wb')  # 打开 wav 文件。
        wf.setnchannels(1)  # 声道设置
        wf.setsampwidth(p.get_sample_size(self.format))  # 采样位数设置
        wf.setframerate(self.sample_rate)  # 采样频率设置
 
        data = wf_i.readframes(self.chunk)  # 读数据
        for i in range(0, int(self.sample_nums /  self.chunk )):
            datao = stream.read(self.chunk)
            stream.write(data)
            wf.writeframes(datao)  # 写入数据
            data = wf_i.readframes(self.chunk)        
        stream.close()
        p.terminate()
        wf.close()

        pass

    #获得波
    def get_data(self,wavfile = 'test.wav'):
        #获取数据
        wf=wave.open(wavfile,"rb")              #打开wav
        p = pyaudio.PyAudio()                   #创建PyAudio对象
        params = wf.getparams()                 #参数获取
        nchannels, sampwidth, self.doc_frame_rate, self.doc_frame_nums = params[:4]
        stream = p.open(format=p.get_format_from_width(sampwidth),
                        channels=nchannels, 
                        rate=self.doc_frame_rate,
                        output=True)            #创建输出流
                                                #读取完整的帧数据到str_data中，这是一个string类型的数据
        str_data = wf.readframes(self.doc_frame_nums)
        self.doc_wave = np.frombuffer(str_data, dtype=np.short)
        wf.close()                              #关闭wave       

    #获得参考波
    def get_refer_data(self,wavfile = 'test.wav'):
        #获取数据
        wf=wave.open(wavfile,"rb")              #打开wav
        p = pyaudio.PyAudio()                   #创建PyAudio对象
        params = wf.getparams()                 #参数获取
        nchannels, sampwidth, self.doc_frame_rate, self.doc_frame_nums = params[:4]
        stream = p.open(format=p.get_format_from_width(sampwidth),
                        channels=nchannels, 
                        rate=self.doc_frame_rate,
                        output=True)            #创建输出流
                                                #读取完整的帧数据到str_data中，这是一个string类型的数据
        str_data = wf.readframes(self.doc_frame_nums)
        self.doc_refer_wave= np.frombuffer(str_data, dtype=np.short)
        f,t,self.refer_table = signal.spectrogram(self.doc_wave,self.doc_frame_rate,nperseg = self.nperseg,noverlap=self.noverlap,nfft = self.nfft)
        wf.close()                              #关闭wave   

    def draw_time(self):
        #时序绘图
        time = np.linspace(0,self.doc_frame_nums/self.doc_frame_rate,self.doc_frame_nums)
        plt.figure()
        plt.plot(time,self.doc_wave)
        plt.xlim(0,0.002)
        plt.show()
        input("输入任何值跳过")
        pass

    def draw_freq(self):
        #频域绘图
        complex_array = np.fft.fft(self.doc_wave)
        fft_freq = np.fft.fftfreq(self.doc_frame_nums, 1/self.doc_frame_rate)
        fft_pow = np.abs(complex_array)
        plt.figure()
        plt.plot(fft_freq[fft_freq>0],fft_pow[fft_freq>0])
        plt.show()
        input("输入任何值跳过")
        pass

    '''
    def draw_taf(self):
        #绘制时频图
        #1. 获得频域总长度
        fft_freq = np.fft.fftfreq(self.doc_frame_nums, 1/self.doc_frame_rate)
        #2. 构建储存表
        table = np.zeros(shape = (self.doc_frame_nums//self.meshframe+1,self.doc_frame_nums//self.meshframe+1))
        time = np.linspace(0,self.doc_frame_nums/self.doc_frame_rate,self.doc_frame_nums//self.meshframe+1)
        freq = np.linspace(0,np.max(fft_freq),self.doc_frame_nums//self.meshframe+1)
        #3. 以每步self.meshframe的速度，向两边扩展self.meshframe//2，做傅里叶变换
        for i in range(0,self.doc_frame_nums,self.meshframe):
            #傅里叶变换
            fft_pow = np.abs(np.fft.fft(self.doc_wave[i:i+self.meshframe//2]))
            fft_freq = np.fft.fftfreq(self.meshframe//2, 1/self.doc_frame_rate)
            fft_pow = fft_pow[fft_freq>0]
            #将数据填入表中
            self.make_table(table,fft_pow,fft_freq[fft_freq>0],self.doc_frame_nums//self.meshframe+1,i)
        #4. 绘图
        plt.pcolormesh(time,freq,table,vmin=0,vmax = np.max(freq)/5)
        plt.show()
        input()
    def make_table(self,table,fft_pow,fft_freq,ysizetable,time):
        mesh = np.linspace(0,np.max(fft_freq),ysizetable)
        for i in range(np.size(fft_freq)):
            for j in range(ysizetable):
                if fft_freq[i]<mesh[j] and fft_pow[i]!=0:
                    table[time//self.meshframe][j]+=fft_pow[i]
                    break        
    '''

    def make_tf(self):
        #f,t,z = signal.stft(self.doc_wave,self.doc_frame_rate,nperseg = 256,noverlap=128,nfft = 256)
        f,t,z = signal.spectrogram(self.doc_wave,self.doc_frame_rate,nperseg = self.nperseg,noverlap=self.noverlap,nfft = self.nfft)
        # plt.pcolormesh(t,f,abs(z))
        #plt.show()
        z_db = 10*np.log10(np.abs(z))
        #plt.specgram(self.doc_wave,Fs = self.doc_frame_rate)
        #plt.show()
        self.show_table(z_db,t,f)
        self.axe_freq =f
        self.axe_times = t
        self.tftable = z

        self.axe_freq_pas = (np.max(f)-np.min(f))/np.size(f)

    def make_td_d2f(self,vit = 343):
        #另一种埖表方式
        tftable = np.abs(self.tftable)
        max_distance = 1
        shape = np.shape(tftable)
        self.axe_distance = np.linspace(-max_distance,max_distance,shape[0]/10)
        tdtable = np.zeros(shape)

        a = np.max(tftable)
        forc = 0
        
        for i in range(shape[1]):
            f_out = self.axe_freq[np.argmax(self.refer_table[:,i])]
            for j in range(shape[0]):
                distance = self.axe_distance[j]
                forc += self.cal_forfreq(i,f_out,distance)
                tdtable[j][i] += forc
        self.show_table(10*np.log10(tdtable),self.axe_times,self.axe_distance)
        self.tdtable = tdtable
        for i in range(shape[1]/10):
            tdtable[:,i] -= tdtable[:,i-1]
            np.maximum(tdtable[:,i],0.0001)
        self.dtdtable = tdtable
        self.show_table(10*np.log10(tdtable),self.axe_times,self.axe_distance,-10,100)
        return 0

    def show_table(self,table,axe_x,axe_y,vmin = 0,vmax = 20):
        plt.pcolormesh(axe_x,axe_y,table)
        plt.show()

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
    
    def filtre_test(self):
        b,a = signal.butter(8,2*1500/self.sample_rate)
        filtedData = signal.filtfilt(b,a,self.doc_wave)
        f,t,z = signal.spectrogram(filtedData,self.doc_frame_rate,nperseg = self.nperseg,noverlap=self.noverlap,nfft = self.nfft)
        #plt.pcolormesh(t,f,10*np.log10(abs(z)))
        #plt.show()
        z = abs(z)

        for i in range(np.shape(z)[1]):
            for j in range(np.shape(z)[0]):
                z[j,i] -= z[j,i-1]

        z = 10*np.log10(abs(z))
        plt.pcolormesh(t,f[f<2000],z[f<2000,:])
        plt.show()
        return 0
f = FMCW()
#f.record_gene()
f.get_data('record.wav')
f.filtre_test()
f.get_refer_data()
f.make_tf()
f.make_td_d2f()
