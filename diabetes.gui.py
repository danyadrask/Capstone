import tkinter as tk
from diabetes_model import *


class DdGui:
    def __init__(self):

        # Create the main window.
        self.main_window = tk.Tk()
        self.main_window.title("Diabetes Prediction Calculator")

# Create frames to group widgets.
        self.one_frame = tk.Frame()
        self.two_frame = tk.Frame()
        self.three_frame = tk.Frame()
        self.four_frame = tk.Frame()
        self.five_frame = tk.Frame()
        self.six_frame = tk.Frame()
        self.seven_frame = tk.Frame()
        self.eight_frame = tk.Frame()
        self.nine_frame = tk.Frame()
        self.ten_frame = tk.Frame()
        self.eleven_frame = tk.Frame()

# Create the widgets for one frame. (title)
        self.title_label = tk.Label(self.one_frame, text='HEART DISEASE PREDICTOR', fg="Blue", font=("Helvetica", 18))
        self.title_label.pack()

# Create the widgets for two frame. (age input)
        self.age_label = tk.Label(self.two_frame, text='Age:')
        self.age_entry = tk.Entry(self.two_frame, bg="white", fg="black", width=10)
        self.age_label.pack(side='left')
        self.age_entry.pack(side='left')

# Create the widgets for three frame. (pregnancy input)
        self.preg_label = tk.Label(self.three_frame, text='Pregnancy:')
        self.preg_entry = tk.Entry(self.three_frame, bg="white", fg="black", width=10)
        self.preg_label.pack(side='left')
        self.preg_entry.pack(side='left')

# Create the widgets for four frame. (blood pressure input)
        self.bp_label = tk.Label(self.four_frame, text='Blood Pressure:')
        self.bp_entry = tk.Entry(self.four_frame, bg="white", fg="black", width=10)
        self.bp_label.pack(side='left')
        self.bp_entry.pack(side='left')

# Create the widgets for five frame. (insulin input)
        self.ins_label = tk.Label(self.five_frame, text='Insulin:')
        self.ins_entry = tk.Entry(self.five_frame, bg="white", fg="black", width=10)
        self.ins_label.pack(side='left')
        self.bp_entry.pack(side='left')

# Create the widgets for six frame. (bmi  input)
        self.bmi_label = tk.Label(self.six_frame, text='BMI:')
        self.bmi_entry = tk.Entry(self.six_frame, bg="white", fg="black")
        self.bmi_label.pack(side='left')
        self.bmi_entry.pack(side='left')

# Create the widgets for seven frame. (fasting blood sugar input)
        self.bldsgr_label = tk.Label(self.seven_frame, text='Blood Sugar:')
        self.bldsgr_entry = tk.Entry(self.seven_frame, bg="white", fg="black")
        self.bldsgr_label.pack(side='left')
        self.bldsgr_entry.pack(side='left')

# Create the widgets for eight frame. (glucose input)
        self.glu_label = tk.Label(self.eight_frame, text='Glucose:')
        self.glu_entry = tk.Entry(self.eight_frame, bg="white", fg="black")
        self.glu_label.pack(side='left')
        self.glu_entry.pack(side='left')

# Create the widgets for nine frame. (pedigree function input)
        self.dpf_label = tk.Label(self.nine_frame, text='Diabetes Pedigree Function')
        self.dpf_entry = tk.Entry(self.nine_frame, bg="white", fg="black")
        self.dpf_label.pack(side='left')
        self.dpf_entry.pack(side='left')

# Create the widgets for ten frame. (skin thickness input)
        self.sknthk_label = tk.Label(self.ten_frame, text='Skin Thickness:')
        self.sknthk_entry = tk.Entry(self.ten_frame, bg="white", fg="black")
        self.sknthk_label.pack(side='left')
        self.sknthk_entry.pack(side='left')

# Create the widgets for fifteen frame = hd (prediction of diabetes)
        self.outcome_predict_ta = tk.Text(self.eleven_frame, height=10, width=25, bg='light blue')

# Create predict button and quit button
        self.btn_predict = tk.Button(self.eleven_frame, text='Predict Diabetes', command=self.predict_outcome)
        self.btn_quit = tk.Button(self.eleven_frame, text='Quit', command=self.main_window.destroy)
        self.outcome_predict_ta.pack(side='left')
        self.btn_predict.pack()
        self.btn_quit.pack()

        self.one_frame.pack()
        self.two_frame.pack()
        self.three_frame.pack()
        self.four_frame.pack()
        self.five_frame.pack()
        self.six_frame.pack()
        self.seven_frame.pack()
        self.eight_frame.pack()
        self.nine_frame.pack()
        self.ten_frame.pack()
        self.eleven_frame.pack()

        tk.mainloop()

    def predict_outcome(self):
        result_string = ""

        self.outcome_predict_ta.delete(0.0, tk.END)
        age = self.age_entry.get()

        blood_pressure = self.bp_entry.get()
        insulin = self.ins_entry.get()
        bmi = self.bmi_entry.get()
        blood_sugar = self.bldsgr_entry.get()
        glucose = self.glu_entry.get()
        diabetes_p_func = self.dpf_entry.get()
        skin_thickness = self.sknthk_entry.get()

        pregnancy = self.preg_entry.get()
        if pregnancy == "Yes":
            pregnancy = 0
        elif pregnancy == "No":
            pregnancy = 1

        result_string += "===Patient Diagnosis=== \n"
        patient_info = (blood_pressure, insulin, bmi, blood_sugar,
                        glucose, diabetes_p_func, skin_thickness, pregnancy, age)

        outcome_prediction = best_model.predict([patient_info])
        disp_string = ("This prediction has an accuracy of:", str(model_accuracy))
        result = outcome_prediction

        if outcome_prediction == [0]:
            result_string = (disp_string, '\n', "0 - You have lower risk of diabetes")
        else:
            result_string = (disp_string, '\n' + "1 - You have higher risk of diabetes")
        self.outcome_predict_ta.insert('1.0', result_string)


DD_GUI = DdGui()
