from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image
import cv2
import csv

class DetectionApp:
    def __init__(self, window, title):
        # tiek izvēlēts YOLOv11s modelis, kas ir specifiski apmācīts ar izveidoto datu kopu
        self.model = YOLO("my_model_60e_2nr.pt")
        self.window = window
        self.window.title(title)
        self.current_frame = 0
        self.filename = None
        self.save_confirmation = None
        self.waiting_notice = None
        self.detect_start_notice = None
        self.video_canvas = None
        self.img_tk = None
        self.video_capture = None
        self.start_message = tk.Label(self.window, text="Izvēlēties failu militārās medicīnas vingrinājumu atpazīšanai!", font=("Arial", 20))
        self.start_message.pack(pady=20)
        self.btn_file_upload = tk.Button(window, text="Pievienot failu", height = 3, width=30, command=self.import_file)
        self.btn_file_upload.pack(side="bottom", padx=50, pady=45)
        self.btn_start_detection = tk.Button(window, text="Sākt atpazīšanu",height = 3, width=30, command=self.save_result)
        self.btn_start_detection.pack(side="bottom", padx=50, pady=10)

    def import_file(self):
        self.filename = filedialog.askopenfilename(title="Izvēlēties video failu", filetypes=[("Video files", "*.mp4 *.avi")])
        self.start_message.destroy()
        if not self.filename:
            self.start_message.destroy()
            if not self.waiting_notice:
                self.waiting_notice = tk.Label(self.window, text="Fails nav izvēlēts, mēģiniet vēlreiz!", font=("Arial", 20))
                self.waiting_notice.pack(pady=50)
            else:
                self.waiting_notice.configure(text="Fails nav izvēlēts, mēģiniet vēlreiz!")
        else:
            self.start_message.destroy()
            if not self.waiting_notice:
                self.waiting_notice = tk.Label(self.window, text="Fails tika veiksmīgi pievienots!", font=("Arial", 20))
                self.waiting_notice.pack(pady=50)
            else:
                self.waiting_notice.configure(text="Fails tika veiksmīgi pievienots!")

    def save_result(self):
        res = tk.messagebox.askyesno(title=None, message="Vai vēlaties saglabāt atpazīšanas rezultātus failā?")
        self.save_confirmation = res
        self.detect_exercises()

    def detect_exercises(self):
        if not self.filename:
            self.start_message.destroy()
            if not self.waiting_notice:
                self.waiting_notice = tk.Label(self.window, text="Fails nav pievienots. Nevar sākt atpazīšanas procesu.", font=("Arial", 20))
                self.waiting_notice.pack(pady=50)
            else:
                self.waiting_notice.configure(text="Fails nav pievienots. Nevar sākt atpazīšanas procesu.")
            return
        self.waiting_notice.destroy()
        self.detect_start_notice = tk.Label(self.window, text="Militāro medicīnas vingrinājumu atpazīšana tiek sākta!", font=("Arial", 20))
        self.detect_start_notice.pack(pady=15)
        self.video_canvas = tk.Label(self.window)
        self.video_canvas.pack()
        self.video_capture = cv2.VideoCapture(self.filename)
        self.current_frame = 0
        self.update_frame()

    # Atsevišķas atpazīšanas procesa daļas pielāgotas no
    # Ultralytics YOLO Docs. Model Prediction with Ultralytics YOLO. Tiešsaiste. Ultralytics, 2023. Pieejams: https://docs.ultralytics.com/modes/predict/. [skatīts 2025-05-26].
    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            self.current_frame += 1
            results_vid = self.model.track(frame, persist=True, conf=0.4)
            for result in results_vid:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    box_coord = box.xywh[0].tolist()
                    prediction_res = [self.current_frame, cls, conf, box_coord[0], box_coord[1], box_coord[2], box_coord[3]]
                    if self.save_confirmation:
                        with open('./testa_scenāriju_rezultāti/prediction_results.csv', 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(prediction_res)
            # Ar video materiāla apstrādi un izvadi ideja smelta no
            # Lietotājs ar lietotājvārdu Moon. Video atskaņošanas programma. Python Forum, 2024. Pieejams: https://python-forum.io/thread-42489-page-2.html. [skatīts 2025-05-26].
            # Paul. Video atskaņošana Tkinter logā. Solarian Programmer, 2018. Pieejams: https://solarianprogrammer.com/2018/04/21/python-opencv-show-video-tkinter-window/. [skatīts 2025-05-26].
            # Pielāgots ar mākslīgā intelekta OpenAI. ChatGPT-4o palīdzību
            frame_res = results_vid[0].plot()
            img_conv = cv2.cvtColor(frame_res, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img_conv).resize((640, 380))
            self.img_tk = ImageTk.PhotoImage(image)
            self.video_canvas.config(image=self.img_tk)
            self.window.after(10, self.update_frame)
        else:
            self.video_capture.release()
            self.video_canvas.destroy()

if __name__ == "__main__":
    main_window = tk.Tk()
    main_window.state('zoomed')
    app = DetectionApp(main_window, "Detect Your Exercise")
    main_window.mainloop()