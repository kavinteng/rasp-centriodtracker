from tkinter import *
from tkinter import messagebox
import os
from threading import Thread

def confirm_yesno(message = 'ยืนยันที่จะปิดโปรแกรมหรือไม่'):
    if messagebox.askyesno(title='confirmation',message=message):
        root.destroy()
        sys.exit(1)

def click_cam(num):
    if num == 1:
        os.system('python person_tracking.py rtsp://admin:888888@192.168.7.50:10554/tcp/av0_0')
    elif num == 2:
        os.system('python person_tracking.py rtsp://admin:888888@192.168.7.60:10554/tcp/av0_0')
    elif num == 3:
        os.system('python person_tracking.py rtsp_subtype')

def click_cam_thread(num):
    t = Thread(target=click_cam, args=(num,))
    t.start()

if __name__ == '__main__':
    root = Tk()
    root.title('Application Controller')
    root.geometry('250x300+0+0')

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