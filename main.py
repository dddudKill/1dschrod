from tkinter import font

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk

eV2J = 1.6021766209e-19
J2eV = 6.241509125493693e+18
hbar = 1.0545718e-34
m0 = 9.109382150000000e-31
to_nm = 1e-9


class OneDimensionalSchrodingerSolver:

    def __init__(self, V, width=10., m=1.0, npts=50, **kwargs):
        V_left = kwargs.get('V_left')
        V_right = kwargs.get('V_right')
        angle = kwargs.get('angle')
        self.x = np.linspace(-width, width, npts)
        if len(kwargs.items()) == 0:
            self.Vx = V(self.x)
        elif len(kwargs.items()) == 1:
            self.Vx = V(self.x, angle)
        elif len(kwargs.items()) == 2:
            self.Vx = V(self.x, V_left, V_right, width)
        elif len(kwargs.items()) == 3:
            self.Vx = V(self.x, V_left, V_right, width, angle)
        self.H = -((hbar/to_nm)**2 / (2 * m * m0)) * self.laplacian() * J2eV + np.diag(self.Vx)
        return

    def plot(self, *args, **kwargs):
        titlestring = kwargs.get('titlestring', "Собственные функции гамильтониана в одномерном случае")
        xstring = kwargs.get('xstring', "Координата (нм)")
        ystring = kwargs.get('ystring', "Энергия (эВ)")
        if not args:
            args = [3]
        x = self.x
        E, U = np.linalg.eigh(self.H)
        h = x[1] - x[0]

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), layout='constrained')

        ax1.plot(x, self.Vx, color='k')
        ax2.plot(x, self.Vx, color='k')

        for i in range(*args):
            ax1.axhline(y=E[i], color='k', ls=":")
            ax1.plot(x, U[:, i] + E[i], label="Ψ{0}(x, E), E{0} = {1:.2}".format(i + 1, abs(E[i])))

            ax2.axhline(y=E[i] / np.sqrt(h), color='k', ls=":")
            ax2.plot(x, U[:, i] ** 2 / np.sqrt(h) + E[i],
                     label="|Ψ{0}(x, E)|^2, E{0} = {1:.2}".format(i + 1, abs(E[i])))

        ax1.set_title(titlestring)
        ax1.set_xlabel(xstring)
        ax1.set_ylabel(ystring)
        ax1.legend()

        ax2.set_title('Плотность вероятности')
        ax2.set_xlabel(xstring)
        ax2.set_ylabel(ystring)
        ax2.set_ylim(ax1.get_ylim())
        ax2.legend()

        return

    def laplacian(self):
        x = self.x
        h = x[1] - x[0]
        n = len(x)
        M = -2 * np.identity(n, 'd')
        for i in range(1, n):
            M[i, i - 1] = M[i - 1, i] = 1
        return M / h ** 2


def infinite_well(x): return x * 0


def finite_well(x, V_left=1., V_right=1., width=10.):
    V = np.zeros(x.size, 'd')
    for i in range(x.size):
        if x[i] < -width / 2:
            V[i] = V_left
        elif x[i] > width / 2:
            V[i] = V_right
        else:
            V[i] = 0
    return V


def oscillator(x): return x ** 2


def triangle_well(x, angle=0.4): return x * angle


def triangle_finite_well(x, V_left, V_right, width, angle=0.04): return finite_well(x, V_left, V_right,
                                                                                    width) + triangle_well(x, angle)


class Application(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.geometry('450x360')
        self.title('Одномерная потенциальная яма')
        self.well_type = tk.IntVar()
        self.well_type.set(0)
        self.well_width = tk.DoubleVar()
        self.well_width.set(10.0)
        self.left_wall_potential = tk.DoubleVar()
        self.left_wall_potential.set(1.0)
        self.right_wall_potential = tk.DoubleVar()
        self.right_wall_potential.set(1.0)
        self.energy_levels = tk.IntVar()
        self.energy_levels.set(3)
        self.effective_mass = tk.DoubleVar()
        self.effective_mass.set(1.0)
        self.angle = tk.DoubleVar()
        self.angle.set(0.05)
        self.dots_amount = tk.IntVar()
        self.dots_amount.set(300)

        self.padx = 2
        self.pady = 2
        self.font1 = font.Font(family="Timew New Roman", size=9, weight="normal", slant="roman")

        self.infinite_well_RB = tk.Radiobutton(self, text='Яма бесконечноей глубины', padx=self.padx, pady=self.pady,
                                               font=self.font1, variable=self.well_type, value=0,
                                               command=lambda: self.set_input_inactive(0))
        self.infinite_well_RB.grid(row=0, column=0, sticky="w", padx=self.padx, pady=self.pady)
        self.finite_well_RB = tk.Radiobutton(self, text='Яма конечной глубины', font=self.font1,
                                             variable=self.well_type, value=1, command=lambda: self.set_input_active(0))
        self.finite_well_RB.grid(row=1, column=0, sticky="w", padx=self.padx, pady=self.pady)
        self.oscillator_RB = tk.Radiobutton(self, text='Осциллятор', font=self.font1, variable=self.well_type, value=2,
                                            command=lambda: self.set_input_inactive(0))
        self.oscillator_RB.grid(row=2, column=0, sticky="w", padx=self.padx, pady=self.pady)
        self.triangle_RB = tk.Radiobutton(self, text='Треугольная яма', font=self.font1, variable=self.well_type,
                                          value=3, command=lambda: self.set_input_inactive(1))
        self.triangle_RB.grid(row=3, column=0, sticky="w", padx=self.padx, pady=self.pady)
        self.triangle_finite_well_RB = tk.Radiobutton(self, text='Наклонная яма с конечной глубиной', font=self.font1,
                                                      variable=self.well_type, value=4,
                                                      command=lambda: self.set_input_active(1))
        self.triangle_finite_well_RB.grid(row=4, column=0, sticky="w", padx=self.padx, pady=self.pady)

        self.well_width_Label = tk.Label(self, text='Ширина ямы (нм)', font=self.font1)
        self.well_width_Label.grid(row=5, column=0, sticky="w", padx=self.padx, pady=self.pady)
        self.well_width_Entry = tk.Entry(self, width=10, font=self.font1, textvariable=self.well_width)
        self.well_width_Entry.grid(row=5, column=1, sticky="w", padx=self.padx, pady=self.pady)

        self.left_wall_potential_Label = tk.Label(self, text='Высота левой стенки (эВ)', font=self.font1)
        self.left_wall_potential_Label.grid(row=6, column=0, sticky="w", padx=self.padx, pady=self.pady)
        self.left_wall_potential_Entry = tk.Entry(self, state='disabled', width=10, font=self.font1,
                                                  textvariable=self.left_wall_potential)
        self.left_wall_potential_Entry.grid(row=6, column=1, sticky="w", padx=self.padx, pady=self.pady)

        self.right_wall_potential_Label = tk.Label(self, text='Высота правой стенки (эВ)', font=self.font1)
        self.right_wall_potential_Label.grid(row=7, column=0, sticky="w", padx=self.padx, pady=self.pady)
        self.right_wall_potential_Entry = tk.Entry(self, state='disabled', width=10, font=self.font1,
                                                   textvariable=self.right_wall_potential)
        self.right_wall_potential_Entry.grid(row=7, column=1, sticky="w", padx=self.padx, pady=self.pady)

        self.energy_levels_Label = tk.Label(self, text='Число энергетических уровней', font=self.font1)
        self.energy_levels_Label.grid(row=8, column=0, sticky="w", padx=self.padx, pady=self.pady)
        self.energy_levels_Spinner = tk.Spinbox(self, from_=1, to=10, width=5, font=self.font1,
                                                textvariable=self.energy_levels)
        self.energy_levels_Spinner.grid(row=8, column=1, sticky="w", padx=self.padx, pady=self.pady)

        self.effective_mass_Label = tk.Label(self, text='Эффективная масса', font=self.font1)
        self.effective_mass_Label.grid(row=9, column=0, sticky="w", padx=self.padx, pady=self.pady)
        self.effective_mass_Entry = tk.Entry(self, width=10, font=self.font1, textvariable=self.effective_mass)
        self.effective_mass_Entry.grid(row=9, column=1, sticky="w", padx=self.padx, pady=self.pady)

        self.angle_Label = tk.Label(self, text='Наклон', font=self.font1)
        self.angle_Label.grid(row=10, column=0, sticky="w", padx=self.padx, pady=self.pady)
        self.angle_Entry = tk.Entry(self, state='disabled', width=10, font=self.font1, textvariable=self.angle)
        self.angle_Entry.grid(row=10, column=1, sticky="w", padx=self.padx, pady=self.pady)

        self.dots_amount_Label = tk.Label(self, text='Число точек', font=self.font1)
        self.dots_amount_Label.grid(row=11, column=0, sticky="w", padx=self.padx, pady=self.pady)
        self.dots_amount_Spinner = tk.Spinbox(self, from_=100, to=4000, increment=50, width=5, font=self.font1,
                                              textvariable=self.dots_amount)
        self.dots_amount_Spinner.grid(row=11, column=1, sticky="w", padx=self.padx, pady=self.pady)

        self.proceed_Button = tk.Button(text="Продолжить", command=self.proceed, padx=self.padx, pady=self.pady)
        self.proceed_Button.grid(row=12, column=3, columnspan=2, padx=self.padx, pady=self.pady)

        self.error_Message = tk.Message(self, text='', font=self.font1)
        self.error_Message.grid(row=6, column=3, rowspan=3)

    def set_input_active(self, is_triangle):
        self.left_wall_potential_Entry.config(state='normal')
        self.right_wall_potential_Entry.config(state='normal')
        if is_triangle == 1:
            self.angle_Entry.config(state='normal')
        else:
            self.angle_Entry.config(state='disabled')

    def set_input_inactive(self, is_triangle):
        self.left_wall_potential_Entry.config(state='disabled')
        self.right_wall_potential_Entry.config(state='disabled')
        if is_triangle == 1:
            self.angle_Entry.config(state='normal')
        else:
            self.angle_Entry.config(state='disabled')

    def proceed(self):
        self.error_Message.config(text='')
        try:
            well_type = self.well_type.get()
            well_width = self.well_width.get()
            left_wall_potential = self.left_wall_potential.get()
            right_wall_potential = self.right_wall_potential.get()
            energy_levels = self.energy_levels.get()
            effective_mass = self.effective_mass.get()
            angle = self.angle.get()
            dots_amount = self.dots_amount.get()
            if angle < 0 or well_width < 0 or left_wall_potential < 0 or right_wall_potential < 0 or effective_mass < 0:
                raise tk.TclError
            if angle >= 1.0:
                raise tk.TclError
        except tk.TclError:
            self.error_Message.config(text='Ошибка ввода!\nПроверьте корректность ввода данных!')
        else:
            if well_type == 0:
                well = OneDimensionalSchrodingerSolver(infinite_well, width=well_width, m=effective_mass,
                                                       npts=dots_amount)
                well.plot(energy_levels, titlestring='Потенциальная яма с бесконечно высокими стенками')
                plt.show()
            elif well_type == 1:
                well = OneDimensionalSchrodingerSolver(finite_well, width=well_width, m=effective_mass,
                                                       npts=dots_amount, V_left=left_wall_potential,
                                                       V_right=right_wall_potential)
                well.plot(energy_levels, titlestring='Потенциальная яма конечной глубины')
                plt.show()
            elif well_type == 2:
                well = OneDimensionalSchrodingerSolver(oscillator, width=well_width, m=effective_mass, npts=dots_amount)
                well.plot(energy_levels, titlestring='Гармонический осциллятор')
                plt.show()
            elif well_type == 3:
                well = OneDimensionalSchrodingerSolver(triangle_well, width=well_width, m=effective_mass,
                                                       npts=dots_amount, angle=angle)
                well.plot(energy_levels, titlestring='Треугольная потенциальная яма')
                plt.show()
            elif well_type == 4:
                well = OneDimensionalSchrodingerSolver(triangle_finite_well, width=well_width, m=effective_mass,
                                                       npts=dots_amount, V_left=left_wall_potential,
                                                       V_right=right_wall_potential, angle=angle)
                well.plot(energy_levels, titlestring='Наклонная потенциальная яма с конечной глубиной')
                plt.show()


if __name__ == '__main__':
    app = Application()
    app.mainloop()
