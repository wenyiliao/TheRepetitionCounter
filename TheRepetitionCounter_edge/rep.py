import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

from tkinter import ttk

# Import matplotlib libraries

import warnings
warnings.filterwarnings('ignore')

from IPython import display
import time
from PIL import Image, ImageDraw, ImageFont,ImageTk

import tkinter as tk
from tkinter import Label


#########
class rep_detect():
    def __init__(self,window,mode):

        canvas.destroy()
        normal.destroy()
        goal.destroy()

        #load model
        self.input_size=192
        self.interpreter = tflite.Interpreter(model_path='/opt/optimus/movenet/movenet.tflite')
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        #ui
        self.window=window
        self.window.title('rep_detection')
        self.window.geometry('800x500')
        self.window.resizable(False, False)


        
        #initial parameter
        self.t=30
        self.pos_series=np.zeros((34,self.t))
        self.num=0
        self.excercise='nan'
        self.count=0
        self.record=''
        self.mode=int(mode)

        self.cap = cv2.VideoCapture(5)
        self.cap.set(3,640) # adjust width
        self.cap.set(4,480) # adjust height


        if mode==1:
            image = Image.open("/home/debian/Documents/movenet/mode1.png")
            bg_image = ImageTk.PhotoImage(image)
            mode1_canvas = tk.Canvas(self.window, width=800, height=500)
            mode1_canvas.place(x=0, y=0)
            mode1_canvas.create_image(0, 0, image=bg_image, anchor="nw")
            
            # Create a button
            image = Image.open("/home/debian/Documents/movenet/overhead_press.png").resize((100,115))
            self.overhead_press_img = ImageTk.PhotoImage(image)
            self.btn_one = tk.Button(window, text="Overhead press", command=self.excercise1,font=('DejaVu Serif', 10),bg="#339966",fg="#c81616",bd=5,image=self.overhead_press_img)
            self.btn_one.place(x=10, y=120 ,width=125,height=110)
            
            image = Image.open("/home/debian/Documents/movenet/side_leg_raise.png").resize((100,115))
            self.side_leg_raise_img = ImageTk.PhotoImage(image)
            self.btn_two = tk.Button(window, text="Side leg raise", command=self.excercise2,font=('DejaVu Serif',10),bg="#339966",fg="#c81616",bd=5,image=self.side_leg_raise_img)
            self.btn_two.place(x=10, y=235 ,width=125,height=110)
            

            self.btn_zero = tk.Button(window, text="Restart", command=self.restart,font=('DejaVu Serif', 10),bg="#339966",fg="#c81616",bd=5)
            self.btn_zero.place(x=37.5, y=360 ,width=70,height=40)        

            self.btn_quit = tk.Button(window, text="Finish", command=self.quit,font=('DejaVu Serif', 10),bg="#339966",fg="#c81616",bd=5)
            self.btn_quit.place(x=37.5, y=450 ,width=70,height=40)  
            
            self.btn_back = tk.Button(window, text="Back", command=self.back_choosemode,font=('DejaVu Serif', 10),bg="#339966",fg="#c81616",bd=5)
            self.btn_back.place(x=37.5, y=405 ,width=70,height=40)  
    
            #show excercise
            self.mark_now=tk.Label(window,text="N/A",font=('DejaVu Serif', 10),bg="#dfbf80",fg="#c81616")
            self.mark_now.place(x=10, y=70,width=125,height=40)
            
            #self.instruction=tk.Label(window,width=20, height=3, font=('DejaVu Serif', 9),bg="#dfbf80",fg="#c81616")
            #self.instruction.place(x=10, y=125 ,width=125,height=115)
            
            # Create a label to display the webcam feed
            self.lmain =tk.Label(window)
            self.lmain.place(x=150, y=10,width=640,height=480)
            
            self.update()
            
        elif mode==2:
            def start():
                self.var1=self.var1.get()
                print(self.var1)
                self.user_goal=self.entry.get()
                if self.var1==1:
                    self.excercise='overhead press'
                elif self.var1==2:
                    self.excercise='side leg raise'
                self.goal_canvas.place_forget()
                self.btn_start.destroy()
                self.btn_quit.destroy()
                self.entry.destroy()
                c2.destroy()
                c1.destroy()

                def give_up():
                    self.cap.release()
                    self.window.destroy()
                    
                    self.stop = time.time()
                    s=self.stop-self.start
                    #record window
                    root_record= tk.Tk()
                    root_record.title('exercise record')
                    root_record.geometry('250x250')
                    image = Image.open("/home/debian/Documents/movenet/fail.png")
                    self.bg_image = ImageTk.PhotoImage(image)
                    canvas = tk.Canvas(root_record, width=250, height=250)
                    canvas.pack()
                    canvas.create_image(0, 0, image=self.bg_image, anchor="nw")
                    self.time_record=tk.Label(root_record,text="time: "+str(round(s,2))+"s",font=('DejaVu Serif', 10),bg="#dfbf80")
                    self.time_record.place(relx=0.5, rely=1.0, anchor="s")
                    fps=self.count/s
                    print(fps)
                    

                #count down page
                
                image = Image.open("/home/debian/Documents/movenet/mode2.png")
                self.bg_image = ImageTk.PhotoImage(image)
                self.mode2_canvas = tk.Canvas(window, width=800, height=500)
                self.mode2_canvas.place(x=0, y=0)
                self.mode2_canvas.create_image(0, 0, image=self.bg_image, anchor="nw")
                
                self.lmain =tk.Label(window)
                self.lmain.place(x=150, y=20 ,width=640,height=480)

                self.left=tk.Label(text=str(int(self.user_goal)-(self.num)),font=('DejaVu Serif', 20),bg="#dfbf80",fg="#c81616")
                self.left.place(x=12.5, y=115,width=120,height=70)

                self.btn_zero = tk.Button(window, text="Restart", command=self.restart,font=('DejaVu Serif', 10),bg="#339966",fg="#c81616",bd=5)
                self.btn_zero.place(x=37.5, y=310 ,width=70,height=40) 
               
                self.btn_quit = tk.Button(window, text="Quit", command=give_up,font=('DejaVu Serif', 10),bg="#339966",fg="#c81616",bd=5)
                self.btn_quit.place(x=37.5, y=400 ,width=70,height=40) 
        
                self.btn_back = tk.Button(window, text="Back", command=self.back_choosemode,font=('DejaVu Serif', 10),bg="#339966",fg="#c81616",bd=5)
                self.btn_back.place(x=37.5, y=355 ,width=70,height=40) 

                self.progressbar=ttk.Progressbar(window, mode='determinate')
                self.progressbar.place(x=150, y=0 ,width=640,height=20) 
                
                self.update()


            #choose goal and target exercise page 
            image = Image.open("/home/debian/Documents/movenet/goal.png")
            bg_image = ImageTk.PhotoImage(image)
            self.goal_canvas = tk.Canvas(self.window, width=800, height=500)
            self.goal_canvas.place(x=0, y=0)
            self.goal_canvas.create_image(0, 0, image=bg_image, anchor="nw")
            
            self.btn_start = tk.Button(window, text="Start", command=start,bg="#339966",font=('DejaVu Serif', 10),fg="#c81616",bd=5)
            self.btn_start.place(x=460, y=375 ,width=70,height=40)
    
            self.btn_quit = tk.Button(window, text="Quit", command=self.quit,bg="#339966",font=('DejaVu Serif', 10),fg="#c81616",bd=5)
            self.btn_quit.place(x=430, y=445 ,width=70,height=40)
            
            self.btn_back = tk.Button(window, text="Back", command=self.back_choosemode,bg="#339966",font=('DejaVu Serif', 10),fg="#c81616",bd=5)
            self.btn_back.place(x=320, y=445 ,width=70,height=40)
                                
            self.entry = tk.Entry(window,bd=3)
            self.entry.place(x=260, y=375 ,width=190,height=40)
            
            def print_selection():
                if var1==1:
                    self.excercise='overhead press'
                elif var1==2:
                    self.excercise='side leg raise'
             
            self.var1 = tk.IntVar()
            
            image = Image.open("/home/debian/Documents/movenet/overhead_press.png").resize((120,175))
            self.overhead_press_img = ImageTk.PhotoImage(image)
            c1 = tk.Radiobutton(window, text='Overhead press',variable=self.var1, value=1,indicatoron=0,bg="#339966",font=('DejaVu Serif', 13),fg="#c81616",bd=7,image=self.overhead_press_img)
            c1.place(x=215, y=115 ,width=175,height=180)
            
            image = Image.open("/home/debian/Documents/movenet/side_leg_raise.png").resize((130,175))
            self.side_leg_raise_img = ImageTk.PhotoImage(image)
            c2 = tk.Radiobutton(window, text='Side leg raise',variable=self.var1, value=2,indicatoron=0,bg="#339966",font=('DejaVu Serif', 13),fg="#c81616",bd=7,image=self.side_leg_raise_img)
            c2.place(x=420, y=115 ,width=175,height=180)


        #self.update()
        self.start = time.time()
        
        
        self.window.mainloop()
    
    def signals(self,keypoints_with_scores_, track, t):
        x=keypoints_with_scores_[0,0,:,1]
        y=keypoints_with_scores_[0,0,:,0]
        new=np.stack((x,y))
        new=new.reshape((1,34),order='F')
        track=np.concatenate((track, new.T), axis=1)
        track=track[:,1:t+1]
            
        self.pos_series=track
           
    def rep_count(self,track,count,excercise):
        
        def lowpass(track_,thres):
            sp=np.fft.fft(track_)
            _,t=track_.shape
            freq = np.fft.fftfreq(t)
            sp[:,abs(freq)>thres]=0
            track_=abs(np.fft.ifft(sp))
            return track_
    
        if excercise=='overhead press':   # l_wrist_y 19/l_eye_y 3/r_wrist_y 21/r_eye_y 5
            track=track[[19,3,21,5],:]
            track=lowpass(track,0.05)
            wrist=np.sum(track[[0,2],:],axis=0)
            eye=np.sum(track[[1,3],:],axis=0)
            relative=(wrist[-1]-eye[-1]<0)
    
            represent_1=np.diff(wrist);represent_2=np.diff(represent_1)
            a=represent_1[-1];b=represent_1[-2];c=represent_2[-1]
            if (a>=0) & (b<=0) & (c>=0):
                if  relative:
                    count=count+1
    
        elif excercise=='side leg raise': #r_hip_y 25 #r_knee_y 29 #l_hip_y 23 #l_knee_y 27
            track=track[[25,29,23,27,24,28,22,26],:]
            track=lowpass(track,0.05)
            knee=np.sum(track[[1,3],:],axis=0)
            r_dx=abs(track[4,:]-track[5,:]);r_dy=abs(track[1,:]-track[0,:]) #knee-hip
            l_dx=abs(track[6,:]-track[7,:]);l_dy=abs(track[3,:]-track[2,:])
            r_theta_radians = np.arctan2(r_dy, r_dx)
            l_theta_radians = np.arctan2(l_dy, l_dx)
            relative=r_theta_radians[-1]<1.1 or l_theta_radians[-1]<1.1
            
            represent_1=np.diff(knee);represent_2=np.diff(represent_1)
            a=represent_1[-1];b=represent_1[-2];c=represent_2[-1]
            if (a>=0) & (b<=0) & (c>=0):
                if  relative:
                    count=count+1 
        self.num=count
        
    def update(self):
        ret, frame = self.cap.read()
        if ret:
            img= cv2.flip(frame, 1)  # Flip the frame horizontally
            img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            #determine excercise
            input_image = Image.fromarray(img).resize((self.input_size,self.input_size))
            input_image = np.expand_dims(input_image, axis=0).astype(np.int32)
        
            
            #input 
            self.interpreter.set_tensor(0, input_image)
            
            #apply the model
            self.interpreter.invoke()
            
            #output
            keypoints_with_scores=self.interpreter.get_tensor(self.output_details[0]['index'])  
        
            self.signals(keypoints_with_scores, self.pos_series, self.t)   #renew signal
            self.rep_count(self.pos_series, self.num, self.excercise)
            
            output_img=Image.fromarray(img)
            #if self.mode==1:
            draw = ImageDraw.Draw(output_img)
            font = ImageFont.truetype('Symbola_hint.ttf', 40)
            draw.text((580,10), str(self.num), font=font,fill=(255, 255, 255))
            output_img=np.array(output_img)
            self.count=self.count+1
        
            img = Image.fromarray(output_img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.lmain.imgtk = imgtk
            self.lmain.configure(image=imgtk)

            if self.mode==2:
                self.progressbar['value']=(self.num/int(self.user_goal))*100
                self.left.config(text=str(int(self.user_goal)-(self.num)))
                if int(self.user_goal)<=self.num:
                    self.cap.release()
                    self.window.destroy()
                    
                    self.stop = time.time()
                    s=self.stop-self.start
                    #record window
                    self.window= tk.Tk()
                    self.window.title('exercise record')
                    self.window.geometry('250x250')
                    
                    image = Image.open("/home/debian/Documents/movenet/success.png")
                    self.bg_image = ImageTk.PhotoImage(image)
                    canvas = tk.Canvas(self.window, width=250, height=250)
                    canvas.pack()
                    canvas.create_image(0, 0, image=self.bg_image, anchor="nw")


                    self.btn_back = tk.Button(self.window, text="back", command=self.back_choosemode,bg="#339966",font=('DejaVu Serif', 10),fg="#c81616",bd=5)
                    self.btn_back.place(relx=0.2, rely=1.0, anchor="s")
                    self.time_record=tk.Label(self.window,text="time: "+str(round(s,2))+"s",font=('DejaVu Serif', 10),bg="#dfbf80")
                    self.time_record.place(relx=0.7, rely=1, anchor="s")
                    
                    fps=self.count/s
                    print(fps)

            self.after= self.lmain.after(30, self.update)
        
    def excercise1(self):
        if self.excercise!='nan':
            self.record=self.record+'\n'+str(self.excercise)+':'+str(self.num)
        self.excercise='overhead press'
        self.mark_now.config(text="overhead press")
        #self.instruction.config(text="wrist over your eye"+'\n')
        self.num=0
        self.lmain.after_cancel(self.after)
        image = Image.open("/home/debian/Documents/movenet/overhead_press.png").resize((100,115))
        self.exercise_img = ImageTk.PhotoImage(image)
        #self.instruction.config(image=self.exercise_img)
        
        
        self.update()
        
    def excercise2(self):
        if self.excercise!='nan':
            self.record=self.record+'\n'+str(self.excercise)+':'+str(self.num)
        self.excercise='side leg raise'
        self.mark_now.config(text="side leg raise")
        
        self.num=0
        self.lmain.after_cancel(self.after)
        image = Image.open("/home/debian/Documents/movenet/side_leg_raise.png").resize((100,115))
        self.exercise_img = ImageTk.PhotoImage(image)
        self.update()
        
    def restart(self):
        self.record=self.record+'\n'+str(self.excercise)+':'+str(self.num)
        self.num=0
        #self.lmain.after_cancel(self.after)

    def back_choosemode(self):
        import sys; import os
        self.cap.release()
        self.window.destroy()
        if hasattr(self, 'lmain'):
            self.lmain.after_cancel(self.after)
        python = sys.executable
        os.execl(python, python, *sys.argv)
        

        
    def quit(self):
        self.cap.release()
        self.window.destroy()
        if hasattr(self, 'lmain'):
            self.lmain.after_cancel(self.after)
        self.record=self.record+'\n'+str(self.excercise)+':'+str(self.num)
        
        self.stop = time.time()
        s=self.stop-self.start
        #record window
        if self.excercise!='nan' and self.mode==1:
            self.window= tk.Tk()
            self.window.title('exercise record')
            self.window.geometry('300x450')
            self.window.config(bg="#dfbf80")
            descrip=tk.Label(self.window,text=self.record,bg="#dfbf80",font=('DejaVu Serif', 10))
            descrip.pack(side='top')
            btn_back = tk.Button(self.window, text="Home",  command=self.back_choosemode,font=('DejaVu Serif', 10),bg="#339966",bd=5)
            btn_back.pack(side='bottom')
            time_record=tk.Label(self.window,text="time: "+str(round(s,2))+"s",font=('DejaVu Serif', 10),bg="#dfbf80")
            time_record.pack(side='bottom')
        
        fps=self.count/s
        print(fps)



########
root = tk.Tk()
root.geometry('800x500')
 
image = Image.open("/home/debian/Documents/movenet/background.png")
bg_image = ImageTk.PhotoImage(image)
canvas = tk.Canvas(root, width=800, height=500)
canvas.grid(row=0, column=0)
canvas.create_image(0, 0, image=bg_image, anchor="nw")
 
 
# Define the functions to be called when the buttons are clicked
def cb_normal():
    app = rep_detect(root,1)
 
def cb_goal():
    app = rep_detect(root,2)

    
 
# Center the buttons using columnspan and sticky options
btn_count_img=ImageTk.PhotoImage(file="/home/debian/Documents/movenet/btn_count.png")
normal = tk.Button(root, text="normal", command=cb_normal,bd=3.5,image=btn_count_img )
normal.place(x=240,y=360, width=130, height=80)


btn_goal_img = ImageTk.PhotoImage(file="/home/debian/Documents/movenet/btn_goal.png")
goal = tk.Button(root, text="goal", command=cb_goal,bd=3.5,image=btn_goal_img)
goal.place(x=450,y=360, width=130, height=80)


 
# Configure the grid columns to expand equally

root.mainloop()
