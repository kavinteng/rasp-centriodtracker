from tkinter import *
from tkinter import messagebox
import os
from threading import Thread
import sys
import datetime
from main import main_process, load_all_model,post_data_out

def confirm_yesno(message = 'ยืนยันที่จะปิดโปรแกรมหรือไม่'):
    if messagebox.askyesno(title='confirmation',message=message):
        root.destroy()
        sys.exit(1)

def click_cam(num):
    if num == 1:
        os.system('python main.py rtsp://admin:888888@192.168.1.50:10554/tcp/av0_0 1')
    elif num == 2:
        os.system('python main.py rtsp://admin:888888@192.168.1.60:10554/tcp/av0_0 2')
    elif num == 3:
        os.system('python main.py rtsp_subtype 3')

def run_process(MODEL_MEAN_VALUES, ageList, genderList, faceNet, ageNet, genderNet, model):
    date = datetime.date.today()
    break_vdo = 0
    while True:
        if os.path.isdir(f'backup_file/{date}') == True:
            if len(os.listdir(f'backup_file/{date}')) > 0:
                for file in os.listdir(f'backup_file/{date}'):
                    file_name = f'backup_file/{date}/{file}'
                    break_vdo = main_process(break_vdo,file_name,MODEL_MEAN_VALUES,ageList,genderList,faceNet,ageNet,genderNet,model)

                    if break_vdo == 2:
                        post_data_out()
                        os.remove(f'{file_name}')
                        break_vdo = 0
        if break_vdo == 1:
            break

def click_cam_thread(num):
    t = Thread(target=click_cam, args=(num,))
    t.start()

def run_process_thread(MODEL_MEAN_VALUES, ageList, genderList, faceNet, ageNet, genderNet, model):
    p = Thread(target=run_process, args=(MODEL_MEAN_VALUES, ageList, genderList, faceNet, ageNet, genderNet, model,))
    p.daemon = True
    p.start()

if __name__ == '__main__':
    MODEL_MEAN_VALUES, ageList, genderList, faceNet, ageNet, genderNet, model = load_all_model()
    root = Tk()
    root.title('Application Controller')
    root.geometry('250x300+0+0')

    run_process_thread(MODEL_MEAN_VALUES, ageList, genderList, faceNet, ageNet, genderNet, model)
    cam1 = Button(root, text="cam1", width=20, bg='red', fg='white', command=lambda num=1: click_cam_thread(num))
    cam1.pack(padx=5, pady=5)
    cam2 = Button(root, text="cam2", width=20, bg='red', fg='white', command=lambda num=2: click_cam_thread(num))
    cam2.pack(padx=5, pady=5)
    cam3 = Button(root, text="cam3", width=20, bg='red', fg='white', command=lambda num=3: click_cam_thread(num))
    cam3.pack(padx=5, pady=5)

    # admin = Button(root, text="ADMIN", width=20, bg='red', fg='white', command=admin_control)
    # admin.pack(padx=5, pady=5, side="bottom")

    root.protocol('WM_DELETE_WINDOW', confirm_yesno)
    # root.attributes('-topmost', True)

    root.mainloop()