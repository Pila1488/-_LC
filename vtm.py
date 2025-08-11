import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
from PyQt5 import QtWidgets
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import menu_vtm
import model_LC_win
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.initializers import GlorotUniform, Zeros
from tensorflow.keras.saving import register_keras_serializable
@register_keras_serializable()
class FixedInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super().__init__(*args, **kwargs)


@register_keras_serializable()
class FixedDense(Dense):
    def __init__(self, *args, **kwargs):
        if 'dtype' in kwargs and isinstance(kwargs['dtype'], dict):
            kwargs['dtype'] = tf.float32
        super().__init__(*args, **kwargs)

custom_objects = {
    'InputLayer': FixedInputLayer,
    'Dense': FixedDense,
    'GlorotUniform': GlorotUniform,
    'Zeros': Zeros
}
class Function_1(QtWidgets.QMainWindow, model_LC_win.Ui_Dialog1):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.back_button.clicked.connect(self.change_wind1)
        self.result_button.clicked.connect(self.func)
        self.ret_menu = None

    def change_wind1(self):
        self.close()
        self.ret_menu = Menu()
        self.ret_menu.show()
    def func(self):

        self.answer_LC.clear()

        try:

            age = int(self.TB_age.toPlainText())

            energy = int(self.TB_energy.toPlainText())
            oxy = int(self.TB_oxygen.toPlainText())
        except:
            self.answer_LC.append("Ошибка: возраст, энергия и кислород должны быть числами")
            return 0

        gender = 0 if self.gender_box.currentText() == 'female' else 1
        smoke = 1 if self.smoking_box.currentText() == 'yes' else 0
        finger = 1 if self.finger_box.currentText() == 'yes' else 0
        mental = 1 if self.mental_box.currentText() == 'yes' else 0
        polut = 1 if self.pollution_box.currentText() == 'yes' else 0
        ill = 1 if self.ill_box.currentText() == 'yes' else 0
        imm = 1 if self.immune_box.currentText() == 'yes' else 0
        breath = 1 if self.breath_box.currentText() == 'yes' else 0
        alco = 1 if self.alcohol_box.currentText() == 'yes' else 0
        thro = 1 if self.throat_box.currentText() == 'yes' else 0
        chest = 1 if self.chest_box.currentText() == 'yes' else 0
        family_cancer = 1 if self.family_box.currentText() == 'yes' else 0
        family_smoke = 1 if self.family_smoke_box.currentText() == 'yes' else 0
        stress_aff = 1 if self.stress_eff_box.currentText() == 'yes' else 0

        new_data = pd.DataFrame({'AGE': [int(age)], 'GENDER': [int(gender)], 'SMOKING': [int(smoke)], 'FINGER_DISCOLORATION': [int(finger)],
                                 'MENTAL_STRESS': [int(mental)], 'EXPOSURE_TO_POLLUTION': [int(polut)], 'LONG_TERM_ILLNESS': [int(ill)],
                                 'ENERGY_LEVEL': [int(energy)],
                                 'IMMUNE_WEAKNESS': [int(imm)], 'BREATHING_ISSUE': [int(breath)], 'THROAT_DISCOMFORT': [int(thro)],
                                 'ALCOHOL_CONSUMPTION': [int(alco)],
                                 'OXYGEN_SATURATION': [int(oxy)], 'CHEST_TIGHTNESS': [int(chest)], 'FAMILY_HISTORY': [int(family_cancer)],
                                 'SMOKING_FAMILY_HISTORY': [int(family_smoke)],
                                 'STRESS_IMMUNE': [int(stress_aff)], 'PULMONARY_DISEASE': [0]})


        try:
            model = load_model('model_LC.h5', custom_objects=custom_objects,
                compile=False)
        except Exception as e:
            self.answer_LC.append(f"Ошибка загрузки модели: {str(e)}")
            return


        x = new_data.drop('PULMONARY_DISEASE', axis=1)

        predictions1 = model.predict(x)
        y_pred1 = predictions1.argmax(axis=1)
        self.answer_LC.append(f'Вероятность отсутствия рака лёгких: {np.asarray(predictions1[0])[0] * 100}%')
        self.answer_LC.append(f'Вероятность наличия рака лёгких: {100 - np.asarray(predictions1[0])[0] * 100}%')
        if np.asarray(y_pred1)[0] == 1:
            self.answer_LC.append('Высокая вероятность рака лёгких')
        if np.asarray(y_pred1)[0] == 0:
            self.answer_LC.append('Низкая вероятность рака лёгких')










class Menu(QtWidgets.QMainWindow, menu_vtm.Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.in_window_model_LC_but.clicked.connect(self.change_wind)
        self.function_window = None

    def change_wind(self):
        self.close()
        self.function_window = Function_1()
        self.function_window.show()




def main():
    app = QtWidgets.QApplication(sys.argv)
    window = Menu()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()