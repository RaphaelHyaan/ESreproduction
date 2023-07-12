import threading
import pyaudio
import wave
import numpy as np

def send_signal(self):
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(2),
                    channels=2,
                    rate=int(self.sample_rate),
                    output=True,
                    frames_per_buffer=self.chunk)
    wf_i = wave.open(self.path_in, 'rb')  # 读 wav 文件

    data = wf_i.readframes(self.chirp_last)  # 读数据
    for i in range(0, self.chirp_nums):
        if i == self.remove_nums:
            print('*')
        stream.write(data)

    stream.close()
    p.terminate()

def receive_signal(self,return_numpy=False):
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(2),
                    channels=2,
                    rate=int(self.sample_rate),
                    input=True,
                    frames_per_buffer=self.chunk)
    wf = wave.open(self.path_out, 'wb')  # 打开 wav 文件。
    wf.setnchannels(2)  # 声道设置
    wf.setsampwidth(p.get_sample_size(self.format))  # 采样位数设置
    wf.setframerate(self.sample_rate)  # 采样频率设置

    data_list = []
    for i in range(0, self.chirp_nums):
        datao = stream.read(self.chirp_last, exception_on_overflow=False)
        wf.writeframes(datao)  # 写入数据
        if return_numpy:
            data_array = np.frombuffer(datao, dtype=np.int16)
            data_list.append(data_array)

    stream.close()
    p.terminate()
    wf.close()

    if return_numpy:
        return np.concatenate(data_list)


def sr_thread():
    t1 = threading.Thread(target=send_signal)
    t2 = threading.Thread(target=receive_signal, kwargs={'return_numpy': True})
    t1.start()
    t2.start()
    t1.join()
    data = t2.join()
    return data