#CTkMessagebox SOURCE INSTALL
#https://github.com/Akascape/CTkMessagebox      

#CustomTkinter Docs
#https://customtkinter.tomschimansky.com/documentation/

#Signal Filtering in Python
#https://swharden.com/blog/2020-09-23-signal-filtering-in-python/

import tkinter
import customtkinter
from CTkMessagebox import CTkMessagebox

import os 
import wave
import pyaudio
import sounddevice as sd
import threading 
import time
#import tkinter as tk
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import scipy.signal
from scipy.fft import *
from scipy.io.wavfile import write ,read
from scipy import fftpack as scfft
from tkinter import filedialog as fd
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
plt.switch_backend('agg')

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


def show_plot(self,wav):
        self.sample_freq = wav.getframerate()
        self.n_samples = wav.getnframes()
        self.t_audio = self.n_samples/self.sample_freq
        self.signal_wave = wav.readframes(self.n_samples)
        self.signal_array = np.frombuffer(self.signal_wave,dtype=np.int16)
        
        
        if wav.getnchannels()==2:            
            CTkMessagebox(title="Información",message="Stereo files are not supported")        
            os.exit(0)
        #
        plt.title("Waveform of Wave File")
        Time=np.linspace(0,self.n_samples/self.sample_freq,num=self.n_samples)
        plt.plot(Time,self.signal_array,color="blue")
        
        plt.ylabel("Amplitude")
        
        #plt.show()
        outputFile = wave.open('output.wav', 'w')
        params = wav.getparams()
        sampleRate = params.framerate
        bufferSize = 1024
        print(params.nframes)
        outputFile.setparams(params)
        samples = []


        
        fig,ax = plt.subplots()
        plt.xlabel="Time(s)"
        fig2,ax2=plt.subplots()
       
        res, = ax.plot(Time,self.signal_array,color="orange")
        res2 = ax2.specgram(self.signal_array,Fs=self.sample_freq,vmin=-20,vmax=50)

        canvas = FigureCanvasTkAgg(fig,self.tabview.tab("Señal original"))
        canvas.draw()
        
        canvas.get_tk_widget().pack(fill=tkinter.BOTH)
        canvas2 = FigureCanvasTkAgg(fig2,self.tabview.tab("Espectro de la señal"))
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tkinter.BOTH,expand=True)
        
class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("Audio Filtering Tool")
        self.geometry(f"{1100}x{580}")

        # configure grid layout
        #https://www.pythontutorial.net/tkinter/tkinter-grid/
        self.grid_columnconfigure((0,3), weight=1)
        self.grid_columnconfigure((1,2), weight=3)        
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure((1, 2, 3), weight=3)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Procesamiento", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 2))

        self.logo_label1 = customtkinter.CTkLabel(self.sidebar_frame, text="Digital", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label1.grid(row=1, column=0, padx=20, pady=(0, 2))

        self.logo_label1 = customtkinter.CTkLabel(self.sidebar_frame, text="de Señales", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label1.grid(row=2, column=0, padx=20, pady=(0, 10))

        self.apply_button = customtkinter.CTkButton(self.sidebar_frame, text="Acerca de",
                                                           command=self.open_input_dialog_event)
        self.apply_button.grid(row=5, column=0, padx=20, pady=(10, 10))                            

        
        

        # create tabview
        self.tabview = customtkinter.CTkTabview(self)
        self.tabview.grid(row=1, column=1, columnspan=2, rowspan=3, padx=(20, 0), pady=(0, 0), sticky="nsew")
        self.tabview.add("Señal original")
        self.tabview.add("Señal Filtrada")
        self.tabview.add("Espectro de la señal")        
        self.tabview.tab("Señal original").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
        self.tabview.tab("Espectro de la señal").grid_columnconfigure(0, weight=1)



        # create radiobutton frame
        self.options_frame = customtkinter.CTkFrame(self)
        self.options_frame.grid(row=0, column=3, rowspan=2, padx=(20, 20), pady=(20, 0), sticky="nsew")              
        self.opciones_title = customtkinter.CTkLabel(master=self.options_frame, text="Opciones de procesamiento:")
        self.opciones_title.grid(row=0, column=0, columnspan=1, padx=10, pady=10, sticky="")

        self.filters_menu = customtkinter.CTkOptionMenu(self.options_frame, dynamic_resizing=False,
                                                        values=["Filtro Pasa Bajas", "Filtro Pasa Altas", "Filtro Pasa Bandas"],
                                                        command=self.setFilter)
        self.filters_menu.grid(row=1, column=0, padx=20, pady=(20, 10))   

        self.apply_button = customtkinter.CTkButton(self.options_frame, text="Aplicar",
                                                           command=self.apply_filter)
        self.apply_button.grid(row=2, column=0, padx=20, pady=(10, 10))

        self.play_result_button = customtkinter.CTkButton(self.options_frame, text="Reproducir Resultado", command=self.open_new_window)
        self.play_result_button.grid(row=3, column=0, padx=20, pady=(10, 10))
                
     

        # create slider and progressbar frame
        self.navbar_frame = customtkinter.CTkFrame(self)
        self.navbar_frame.grid(row=0, column=1, rowspan=1, columnspan=2, padx=(20, 20), pady=(20, 0), sticky="n")
        self.navbar_frame.grid_columnconfigure(1, weight=3)
        self.navbar_frame.grid_columnconfigure((0,2,3), weight=0)
        self.navbar_frame.grid_rowconfigure(0, weight=0)


        self.labelfreq = customtkinter.CTkLabel(self.navbar_frame, text="Frecuencia de corte", font=customtkinter.CTkFont(size=10, weight="bold"))
        self.labelfreq.grid(row=0, column=0, columnspan=1, padx=10, pady=10, sticky="")

        self.slider_1 = customtkinter.CTkSlider(self.navbar_frame, from_=0.01, to=0.99, number_of_steps=100, command=self.slider_changed)
        self.slider_1.grid(row=0, column=1, padx=(20, 10), pady=(10, 10), sticky="ew")

        self.cutOffFreq = customtkinter.CTkLabel(self.navbar_frame, text="0.500", font=customtkinter.CTkFont(size=10, weight="bold"))
        self.cutOffFreq.grid(row=0, column=2, padx=20, pady=(0, 2))



        self.labelfreq2 = customtkinter.CTkLabel(self.navbar_frame, text="Frec. de corte Inferior", font=customtkinter.CTkFont(size=10, weight="bold"))
        

        self.slider_2 = customtkinter.CTkSlider(self.navbar_frame, from_=0.01, to=0.99, number_of_steps=100, command=self.slider2_changed)
        

        self.cutOffFreq2 = customtkinter.CTkLabel(self.navbar_frame, text="0.500", font=customtkinter.CTkFont(size=10, weight="bold"))
        



        self.record_button = customtkinter.CTkButton(master=self.navbar_frame, width=10, height=10, text="Grabar Audio", fg_color="red", text_color=("gray10", "#DCE4EE"), command=self.click_handler)        
        self.record_button.grid(row=0, column=3, padx=(0, 0), pady=(20, 20), sticky="nsew")        

        self.openFile_button = customtkinter.CTkButton(master=self.navbar_frame, width=10, height=10, text="Abrir archivo", fg_color="red", text_color=("gray10", "#DCE4EE"), command=self.select_file)        
        self.openFile_button.grid(row=1, column=3, padx=(0, 0), pady=(20, 20), sticky="nsew")        

        self.labeltime = customtkinter.CTkLabel(self.navbar_frame, text="00:00:00", font=customtkinter.CTkFont(size=10, weight="bold"))
        self.labeltime.grid(row=0, column=4, padx=20, pady=(0, 2))
                        
        

        # create checkbox and switch frame
        self.theme_frame = customtkinter.CTkFrame(self)
        self.theme_frame.grid(row=2, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")

        self.appearance_mode_label = customtkinter.CTkLabel(self.theme_frame, text="Apariencia de la aplicación:", anchor="w")
        self.appearance_mode_label.grid(row=1, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.theme_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=2, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.theme_frame, text="Escala:", anchor="w")
        self.scaling_label.grid(row=3, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.theme_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=4, column=0, padx=20, pady=(10, 20))

        

        # set default values              
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")        
        self.filters_menu.set("Filtros")
        self.appearance_mode_optionemenu.set("Tema")
        self.recording = False
        self.currentFilter = ""  

        self.set_passband_widgets()        
    
    def select_file(self):
        filetypes = (
            ('text files', '*.wav'),
            ('All files', '*.*')
        )
    
        filename = fd.askopenfilename(
            title='Open a file',
            initialdir='/',
            filetypes=filetypes)
        try:
            
            wav = wave.open(filename,"r")
            show_plot(self, wav)
        except:
            CTkMessagebox(title="Warning",message="El archivo seleccionado tiene un formato no valido") 


    def open_new_window(self):
        self.toplevel = customtkinter.CTkToplevel()  # master argument is optional 
        self.toplevel.title('Audio filtrado')
        self.toplevel.geometry("300x200")
        self.toplevel.grid_columnconfigure((0,2), weight=1)        
        self.toplevel.grid_rowconfigure(1, weight=3)      

        IMAGE_WIDTH = 100
        IMAGE_HEIGHT = 100
        IMAGE_PATH = 'wave.jpg'

        your_image = customtkinter.CTkImage(light_image=Image.open(os.path.join(IMAGE_PATH)), size=(IMAGE_WIDTH , IMAGE_HEIGHT))
        label = customtkinter.CTkLabel(master=self.toplevel, image=your_image, text='')
        label.grid(row=0, column=1, padx=(0, 0), pady=(20, 20), rowspan=1, sticky="nsew")

        self.toplevel.play_button = customtkinter.CTkButton(master=self.toplevel, width=10, height=10, text="Reproducir", command=self.play_audio)        
        self.toplevel.play_button.grid(row=1, column=1, padx=(0, 0), pady=(20, 20), rowspan=1, sticky="nsew")  
        self.toplevel.resizable(False, False)
        
        
        
    def play_audio(self):
        wav = wave.open('New-Filtered.wav',"r")
        self.signal_wave = wav.readframes(self.n_samples)
        self.filtered_signal = np.frombuffer(self.signal_wave,dtype=np.int16)
        sample_rate = 44100  # Sample rate of the audio data        
        sd.play(self.filtered_signal, sample_rate)
        sd.wait()                                  
               

   
     
    def slider_changed(self, event):
        self.cutOffFreq.configure(text=f"{self.slider_1.get():.3f}") 
    
    def slider2_changed(self, event):
        self.cutOffFreq2.configure(text=f"{self.slider_2.get():.3f}") 

    def open_input_dialog_event(self):              
        CTkMessagebox(title="Acerca de", message="\nFacultad de Ingeniería, UASLP\nÁrea de Ciencias de la Computación\n\nInterfaces de comunicaciones\nProyecto de la materia\n\nDesarrollado por:\nVázquez Garcia Daniel Alejandro\nIpiña Zarazúa José Alfredo\n")        
    
    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    #Eliminar widgets del frame, se usa para actualizar la grafica
    def clear_frame(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

    def set_passband_widgets(self):
        if self.currentFilter=="Filtro Pasa Bandas":
            self.labelfreq2.grid(row=1, column=0, columnspan=1, padx=10, pady=10, sticky="")
            self.slider_2.grid(row=1, column=1, padx=(20, 10), pady=(10, 10), sticky="ew")
            self.cutOffFreq2.grid(row=1, column=2, padx=20, pady=(0, 2))
        else:
            self.labelfreq2.grid_forget() 
            self.slider_2.grid_forget() 
            self.cutOffFreq2.grid_forget() 
               
        
    def switch(self, filter):
        if filter == "Filtro Pasa Bajas":                                                                     
            return self.butter_lowpass(self.signal_array, self.cutOffFreq.cget("text"))
        elif filter == "Filtro Pasa Altas":
            return self.butter_highpass(self.signal_array, self.cutOffFreq.cget("text"))
        elif filter == "Filtro Pasa Bandas":
            return self.butter_bandpass(self.signal_array, self.cutOffFreq2.cget("text"), self.cutOffFreq.cget("text")) #inferior, superior     
    
    def butter_lowpass(self, signal, cutOff):
        
        b, a = scipy.signal.butter(3, float(cutOff), 'lowpass')
        #b, a = scipy.signal.butter(3, float(fc), 'lowpass')
        y = scipy.signal.filtfilt(b, a, signal)
        return y
    
    def butter_highpass(self, signal, cutOff):
        b, a = scipy.signal.butter(3, float(cutOff), 'highpass')
        y = scipy.signal.filtfilt(b, a, signal)
        return y
    
    def butter_bandpass(self, signal, lowcut, highcut):
        #fs=5000
        #order=5
        #nyq = 0.5 * fs   
          
        #low = lowcut / nyq
        #high = highcut / nyq
        b, a = scipy.signal.butter(3, [float(lowcut), float(highcut)], btype='band')
        y = scipy.signal.lfilter(b, a, signal)
        return y

    def bandPassFilter(signal,fs):
         fs=fs
         lowcut = 20.0
         highcut = 50.0
         nyq = 0.5 * fs
         low = lowcut/nyq
         high = highcut/nyq
         
         order = 2
         b,a = scipy.signal.butter(order,[low,high],'bandpass',analog=False)
         y = scipy.signal.filtfilt(b,a,signal,axis=0)
         return (y)

    def setFilter(self, newFilter: str):
        self.currentFilter = newFilter
        if(newFilter == "Filtro Pasa Bandas"):            
            self.labelfreq.configure(text='Frec. de Corte Superior')
            self.set_passband_widgets()
        else:
            self.labelfreq.configure(text='Frecuencia de corte')
            self.set_passband_widgets()
            


    def apply_filter(self):
        if self.currentFilter == "Filtro Pasa Bandas" and float(self.cutOffFreq.cget("text")) < float(self.cutOffFreq2.cget("text")):
            CTkMessagebox(title="Información",message="La frecuencia de corte superior no puede ser menor a la inferior\nModifique los valores")   
        else:
            if(self.currentFilter != ""):
                self.clear_frame(self.tabview.tab("Señal Filtrada"))
                y = self.switch(self.currentFilter)
                self.filtered_signal = y
                fig,ax=plt.subplots()
                Time=np.linspace(0,self.n_samples/self.sample_freq,num=self.n_samples)
                res, = ax.plot(Time,y,color="Blue")            

                canvas = FigureCanvasTkAgg(fig,self.tabview.tab("Señal Filtrada"))
                canvas.draw()            
                canvas.get_tk_widget().pack(fill=tkinter.BOTH)
                            
                self.tabview.set('Señal Filtrada')
                samplerate=44100
                write("New-Filtered.wav",  samplerate, y.astype(np.int16))
            else:
                CTkMessagebox(title="Información",message="Seleccione un filtro para aplicar")        



    def click_handler(self):        
        if self.recording:
            self.recording = False  
            self.record_button.configure(text="Grabar Audio")            
        else:
            self.recording = True     
            self.record_button.configure(text="Detener grabación")       
            threading.Thread(target=self.record).start()
        

    
    def record(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16,channels=1,rate=44100,input=True,frames_per_buffer=1024)
        
        frames = []
        
        start = time.time()
        
        while self.recording:
            data = stream.read(1024)
            frames.append(data)
            
            passed = time.time()-start
            secs = passed % 60
            mins = passed // 60
            hours = mins//60
            
            self.labeltime.configure(text=f"{int(hours):02d}:{int(mins):02d}:{int(secs):02d}")
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        exists = True
        i = 1
        while exists:
            if os.path.exists(f"recording{i}.wav"):
                i+= 1
            else:
                exists=False
                
        sound_file = wave.open(f"recording{i}.wav","wb")
        file = "recording",i,".wav"
        sound_file.setnchannels(1)
        sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(b"".join(frames))
        sound_file.close()
        
        
        wav = wave.open(f"recording{i}.wav","r")
        Fs=44100
        n_fft=1024
        f_hertz = np.fft.rfftfreq(n_fft, 1 / Fs) 
        sr, data = wavfile.read(f"recording{i}.wav")
        # Fourier Transform
        N = len(data)
        yf = rfft(data)
        xf = rfftfreq(N, 1 / sr)
        
        
        
         # Get the most dominant frequency and return it
        idx = np.argmax(np.abs(yf))
        print("Max freq",idx)
        freq = xf[idx]
        print("Freq",freq)
        show_plot(self,wav)
        

        

        


if __name__ == "__main__":
    app = App()
    app.mainloop()