import tkinter as tk
from tkinter import ttk
import numpy as np
import math as m
from scipy.linalg import fractional_matrix_power


def to_tuple(array, offset=None):
    if isinstance(offset, np.ndarray):
        array += offset
    return array[0, 0], array[1, 0], array[0, 1], array[1, 1]


def to_array(sequence, offset=None):
    arr = np.array([[sequence[0], sequence[2]], [sequence[1], sequence[3]]])
    if isinstance(offset, np.ndarray):
        return arr - offset
    return arr


class Vector:
    base_colours = ['lawn green', 'red3']

    def __init__(self, gridob, matrix, nature):
        self.gridob = gridob
        self.canvas = gridob.canvas
        self.matrix = matrix
        self.canvas_id = self.plot_vector(matrix, nature)

    def plot_vector(self, matrix, nature):
        state = tk.NORMAL
        tag = 'vector'
        if nature == 'base':
            colour = Vector.base_colours.pop(0)
        elif nature == 'e':
            colour = 'yellow2'
            state = tk.HIDDEN
            tag = ('vector', 'eigen')
        else:
            colour = 'blue'
        coords = self.get_coords(matrix)
        v = self.canvas.create_line(coords, arrow=tk.LAST, fill=colour, width=3, tags=tag, state=state)
        self.gridob.canvas_dictionary[v] = coords
        return v

    def get_coords(self, matrix):
        x1 = self.gridob.zero_matrix[0, 0]
        y1 = self.gridob.zero_matrix[1, 0]
        x2 = x1 + (matrix[0, 0] * self.gridob.space)
        y2 = y1 - (matrix[1, 0] * self.gridob.space)
        return x1, y1, x2, y2


class Grid:
    def __init__(self, master, matrix_data, eigvars):
        self.canvas = master
        self.data_source = matrix_data
        self.base_count = 6
        self.display_count = self.base_count
        self.space = self.update_space()
        self.zero_matrix = self.get_zero_matrix()
        self.inc_scale_factor = 0
        self.step_matrix = 0
        self.canvas_dictionary = {}
        self.vector_stack = np.array([[], []])
        self.max_outlier = ((self.display_count - 1) / 2)
        self.eigvars = eigvars
        self.trace_colour = 'pink'
        self.trace_width = 1

        self.plot_squared_gridlines(color=self.trace_colour, start_index=0, tag='trace', width=self.trace_width)
        self.plot_squared_gridlines(start_index=0, tag='grid')
        self.plot_vectors(self.data_source, 'base')

    def update_space(self):
        self.space = ((self.canvas.winfo_reqheight() - 6) / self.display_count)
        return self.space

    def get_zero_matrix(self):
        zpoint = self.canvas.winfo_reqwidth() / 2, self.canvas.winfo_reqheight() / 2
        return np.array([[zpoint[0], zpoint[0]], [zpoint[1], zpoint[1]]])

    def plot_squared_gridlines(self, tag, start_index=0, color='gray82', width=2):
        end_point = m.ceil(self.display_count/2)
        # rounding up so that a nice square is always made
        #end_index = int(end_point)

        xcentre = self.zero_matrix[0, 0]
        ycentre = self.zero_matrix[1, 0]
        for base_index in range(start_index, int(end_point)+1):
            if base_index == 0:
                index_list = [base_index]
            else:
                index_list = [base_index, -base_index]
            for index in index_list:
                self.create_column(tag, index, end_point, xcentre, ycentre, color, width)
                self.create_row(tag, index, end_point, xcentre, ycentre, color, width)

    def create_column(self, tag, index, end_point, xcen, ycen, color, width):
        x1 = x2 = xcen + self.space * index
        y1 = ycen - end_point * self.space
        y2 = ycen + end_point * self.space
        n = self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width, tags=(tag, 'column'))
        self.canvas_dictionary[n] = x1, y1, x2, y2

    def create_row(self, tag, index, end_point, xcen, ycen, color, width):
        y1 = y2 = ycen + index * self.space
        x1 = xcen - end_point * self.space
        x2 = xcen + end_point * self.space
        n = self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width, tags=(tag, 'row'))
        self.canvas_dictionary[n] = x1, y1, x2, y2

    def plot_vectors(self, source, nature):
        plot_matrix = source.mvalues()
        if nature == 'e':
            values, vectors = np.linalg.eig(plot_matrix)
            for eig_list, var_list in zip((values, vectors), self.eigvars):
                if eig_list.shape == (2, 2):
                    eig_list = vectors.tolist()[0] + vectors.tolist()[1]
                for var, value in zip(var_list, eig_list):
                    try:
                        var.set(round(value, 2))
                    except TypeError:
                        value = complex(round(value.real), round(value.imag))
                        var.set(value)

            plot_matrix = vectors
            for row in plot_matrix:
                for val in row:
                    if isinstance(val, complex):
                        return

        self.vector_stack = np.hstack([self.vector_stack, plot_matrix])
        for col_count in range(len(plot_matrix.T)):
            Vector(self, plot_matrix[:, col_count:col_count + 1], nature)

    def transform_handler(self, exponent, slide, det):
        # Need to clear eigenvalues if the matrix changes without using reset
        if not self.inc_scale_factor or exponent == 0:
            new_matrix = self.data_source.mvalues()
            self.inc_scale_factor = self.get_inc_scale(new_matrix, slide)
            new_matrix[0, 1], new_matrix[1, 0] = -new_matrix[0, 1], -new_matrix[1, 0]
            self.step_matrix = fractional_matrix_power(new_matrix, 1 / slide) / self.inc_scale_factor
            if not self.canvas.find_withtag('eigen'):
                self.plot_vectors(self.data_source, 'e')

        self.canvas_transform(['grid', 'vector'], exponent, self.step_matrix, det)
        self.dynamic_grid_update(['trace'], exponent, self.inc_scale_factor)

    def canvas_transform(self, tags, multiplier, matrix, det_var):
        """Applied the incremental matrix to plotted coordinates the specified number of times."""
        the_transform = fractional_matrix_power(matrix, multiplier)
        for tag in tags:
            for item in self.canvas.find_withtag(tag):
                self.actual_transformation(item, the_transform)

        det_var.set(round(np.linalg.det(np.real(fractional_matrix_power(matrix*self.inc_scale_factor, multiplier))), 2))

    def dynamic_grid_update(self, tags, exponent, scale_factor):
        # Adding New Lines:
        self.display_count = self.base_count * scale_factor ** exponent
        difference = ((int(self.display_count + 1)) - len(self.canvas.find_withtag('trace'))//2)
        # this describes the difference between the lines to be displayed on one one column or row, and the number that are currently displaced
        # should always be an even number of trace lines as the trace will always be square
        # the '+1' accounts for the centre line, which is not counted by the display count (which counts spaces)
        if difference > 1:
            start = int((len(self.canvas.find_withtag('trace'))-2) / 4) + 1
            self.plot_squared_gridlines(tag='trace', start_index=start, color=self.trace_colour, width=self.trace_width)
            self.canvas.tag_lower('trace', 'grid')

        # Scaling Existing Plots
        for tag in tags:
            for trace_index in self.canvas.find_withtag(tag):
                if trace_index in self.canvas.find_withtag('row'):
                    matrix = np.array([[1, 0], [0, scale_factor ** -1]])
                else:
                    matrix = np.array([[scale_factor ** -1, 0], [0, 1]])
                the_transform = fractional_matrix_power(matrix, exponent)
                self.actual_transformation(trace_index, the_transform)

    def actual_transformation(self, item, matrix):
        """Offsets and transforms sets of coordinates"""
        coord_m = to_array(self.canvas_dictionary[item], offset=self.zero_matrix)
        new_coord_m = matrix.dot(coord_m)
        new_coord_m_real = np.real(new_coord_m)
        new_coords = to_tuple(new_coord_m_real, offset=self.zero_matrix)
        self.canvas.coords(item, new_coords)

    def submit_handler_new(self, vector_data):
        """Plots new vectors and adjusts scale for visibility. Note: It assumes that scale position is 0.
        this is accounted for then corrected for by the Visualiser object"""
        pre_vector_max = abs(vector_data.mvalues()).max()
        if pre_vector_max > self.max_outlier:
            scale_factor = pre_vector_max / self.max_outlier
            self.static_grid_update(scale_factor)
            self.max_outlier = pre_vector_max
        self.plot_vectors(vector_data, 'vector')
        #self.inc_scale_factor = 0  # needed?
        #self.step_matrix = 0

    def static_grid_update(self, scale_factor):
        """Recalibrates the grid display to allow full vector lengths to be displayed.
        Assumes operating at 0 slide position."""
        self.display_count = scale_factor * self.base_count
        self.update_space()
        correction = int(m.ceil(self.display_count)) / self.display_count
        self.display_count = int(m.ceil(self.display_count))

        start = self.base_count//2 + 1
        self.base_count = self.display_count

        # Changing length of existing lines
        self.canvas.scale('row', self.zero_matrix[0, 0], self.zero_matrix[1, 0], correction, scale_factor ** -1)
        self.canvas.scale('column', self.zero_matrix[0, 0], self.zero_matrix[1, 0], scale_factor ** -1, correction)
        self.canvas.scale('vector', self.zero_matrix[0, 0], self.zero_matrix[1, 0], scale_factor ** -1, scale_factor ** -1)

        # Adding new lines as required
        self.plot_squared_gridlines(tag='trace', start_index=start, color=self.trace_colour, width=self.trace_width)
        self.plot_squared_gridlines(tag='grid', start_index=start)

        # Logs coordinates for 0 level of transformation. This is important for retaining accuracy and consistency
        for id in self.canvas.find_all():
            self.canvas_dictionary[id] = self.canvas.coords(id)

        self.canvas.tag_lower('trace', 'grid')

    def get_inc_scale(self, matrix, slide):
        """Gets the maximum vector size after the transformation and subsequent scale. this is then
         returned, to the mathematical root of the slide length, so that it can be applied incrementally."""
        trns_max = abs(matrix.dot(self.vector_stack)).max()
        total_scale = trns_max / self.max_outlier
        if total_scale > 1:
            inc_scale = total_scale ** (1 / slide)
        else:
            inc_scale = 1
        return inc_scale

    def toggle_trace(self, var):
        """Toggles visibility of the trace grid"""
        if var.get() == 0:
            self.canvas.itemconfig('trace', state=tk.HIDDEN)
        else:
            self.canvas.itemconfig('trace', state=tk.NORMAL)


class GridContainer(tk.Frame):

    def __init__(self, master, *args, **kwargs):  # If I specify args, I need to take *args out
        super().__init__(master, *args, **kwargs)
        canvas_height = 600
        canvas_width = 600

        self.canvas = tk.Canvas(self, bg='white', height=canvas_height, width=canvas_width)
        self.canvas.grid(row=1, column=0, pady=15, sticky=tk.E)

    def create_display_grid(self, data_source, eigvars):
        self.display_grid = Grid(self.canvas, data_source, eigvars)
        return self.display_grid

    def reset(self, data_source, eigvars):
        Vector.base_colours = ['lawn green', 'red3']
        self.canvas.delete('all')
        del self.display_grid
        self.display_grid = Grid(self.canvas, data_source, eigvars)
        return self.display_grid


class DataSubContainer(tk.Frame):

    def __init__(self, master, cols, rows, set_base=False, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.bases = [tk.DoubleVar() for x in range(cols * rows)]

        if set_base:
            self.set_base()

        self.base = np.array([self.bases[:((cols * rows) // 2)], self.bases[((cols * rows) // 2):]])
        self.create_entries()

    def set_base(self):
        for co, value in zip(self.bases, [1, 0, 0, 1]):#):
            co.set(value)

    def create_entries(self):
        top_row_lab_grid = [(1, x) for x in range(1, int((len(self.bases) / 2) + 1))]
        bot_row_lab_grid = [(2, x) for x in range(1, int((len(self.bases) / 2) + 1))]
        top_row_b = self.bases[0:len(self.bases) // 2]
        bot_row_b = self.bases[len(self.bases) // 2:]
        for varl, varc in zip([top_row_b, bot_row_b], [top_row_lab_grid, bot_row_lab_grid]):
            for var, co in zip(varl, varc):
                entry = ttk.Entry(self, textvariable=var, width=5)
                entry.grid(row=co[0], column=co[1], pady=3, padx=2)

    def mvalues(self):
        row1 = [x.get() for x in self.base[0]]
        row2 = [y.get() for y in self.base[1]]
        return np.array([row1, row2])


class DataLabelFrame(tk.LabelFrame):

    def create_data_input(self, cols, rows):
        dataf = DataSubContainer(self, cols=cols, rows=rows, set_base=True)
        dataf.grid(row=1, column=1, columnspan=2, sticky=tk.W)
        return dataf


class ReadoutLabelFrame(tk.LabelFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        eig_val1 = tk.IntVar()
        eig_val2 = tk.IntVar()
        eig_vectsr1c1 = tk.IntVar()
        eig_vectsr1c2 = tk.IntVar()
        eig_vectsr2c1 = tk.IntVar()
        eig_vectsr2c2 = tk.IntVar()

        self.eigs = [(eig_val1, eig_val2), (eig_vectsr1c1, eig_vectsr1c2, eig_vectsr2c1, eig_vectsr2c2)]

        # Titles
        eig_val_lab1 = tk.Label(self, text='Values:', justify=tk.LEFT, font=('Technic', 10))
        eig_val_lab2 = tk.Label(self, text='Vectors:', justify=tk.LEFT, font=('Technic', 10))

        # Information
        eig_val_1 = tk.Label(self, textvariable=eig_val1, justify=tk.LEFT, font=('Technic', 10))
        eig_comma = tk.Label(self, text=',', justify=tk.LEFT, font=('Technic', 10))
        eig_val_2 = tk.Label(self, textvariable=eig_val2, justify=tk.LEFT, font=('Technic', 10))

        eig_vect_r1c1 = tk.Label(self, textvariable=eig_vectsr1c1, justify=tk.LEFT, font=('Technic', 10))
        eig_vect_r1c2 = tk.Label(self, textvariable=eig_vectsr1c2, justify=tk.LEFT, font=('Technic', 10))
        eig_vect_r2c1 = tk.Label(self, textvariable=eig_vectsr2c1, justify=tk.LEFT, font=('Technic', 10))
        eig_vect_r2c2 = tk.Label(self, textvariable=eig_vectsr2c2, justify=tk.LEFT, font=('Technic', 10))

        # row 1
        eig_val_lab1.grid(row=0, column=1, columnspan=3)
        eig_val_lab2.grid(row=0, column=4, columnspan=2)

        # row 2
        eig_val_1.grid(row=1, column=1)
        eig_comma.grid(row=1, column=2)
        eig_val_2.grid(row=1, column=3)
        eig_vect_r1c1.grid(row=1, column=4, padx=5)
        eig_vect_r1c2.grid(row=1, column=5, padx=5)

        # row 3
        eig_vect_r2c1.grid(row=2, column=4, padx=5)
        eig_vect_r2c2.grid(row=2, column=5, padx=5)


class Visualiser(tk.Frame):

    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        slide = 200
        trace_var = tk.IntVar()
        trace_var.set(1)
        self.det = tk.IntVar()

        datacontainer1 = DataLabelFrame(self, text='Matrix', font=('Technic', 12))
        datacontainer2 = DataLabelFrame(self, text='Vector', font=('Technic', 12))
        eigenframe = ReadoutLabelFrame(self, text='Eigen', font=('Technic', 12))
        det_frame = tk.LabelFrame(self, text='Determinant', font=('Technic', 12))
        det_label = tk.Label(det_frame, textvariable=self.det, justify=tk.LEFT, font=('Technic', 12))
        gridcontainer = GridContainer(self)

        submit_button = tk.Button(self, text='Submit Vect', font=('Technic', 10),
                                   command=lambda: self.button_handler(slider, vector_data, nature='vector'))
        eig_button = tk.Button(self, text='Show Eigen', font=('Technic', 10),
                                command=lambda: self.button_handler(slider, matrix_data, nature='e'))
        reset_button = tk.Button(self, text='Reset', font=('Technic', 10),
                                  command=lambda: self.reset(matrix_data, vector_data, slider, gridcontainer, eigenframe.eigs))
        slider = ttk.Scale(self, orient='horizontal', from_=0, to=slide,
                           length=300, command=lambda x: self.display_grid.transform_handler(float(x),
                                                                                             slide=slide, det=self.det))
        trace_check = ttk.Checkbutton(self, text='Trace', variable=trace_var,
                                      command=lambda: self.display_grid.toggle_trace(trace_var))

        # Row 1
        datacontainer1.grid(row=0, column=0, columnspan=1, sticky=tk.W, padx=2)
        datacontainer2.grid(row=0, column=1, columnspan=1, sticky=tk.W, padx=2, ipadx=2)

        submit_button.grid(row=0, column=2, sticky=tk.NW, pady=6, padx=0)
        eig_button.grid(row=0, column=2, sticky=tk.SW, pady=20, padx=0)

        eigenframe.grid(row=0, column=3, sticky=tk.W, padx=2, columnspan=3)

        # Row 2
        gridcontainer.grid(row=1, column=0, columnspan=6)

        # Row 3
        reset_button.grid(row=2, column=0, pady=6, sticky=tk.NSEW)
        slider.grid(row=2, column=2, columnspan=3)
        det_frame.grid(row=2, column=5, sticky=tk.NE, padx=5, columnspan=1)
        det_label.pack()

        matrix_data = datacontainer1.create_data_input(cols=2, rows=2)
        vector_data = datacontainer2.create_data_input(cols=1, rows=2)
        self.display_grid = gridcontainer.create_display_grid(matrix_data, eigenframe.eigs)

        for co, value in zip(matrix_data.bases, [2, 3, 2, 1]):
            co.set(value)

        # Row 4
        trace_check.grid(row=3, column=0, columnspan=1)

    def button_handler(self, slider, data, nature):
        slide_value = slider.get()
        slider.set(0)
        self.display_grid.inc_scale_factor = 0

        if nature == 'vector':
            self.display_grid.submit_handler_new(data)
        elif nature == 'e':
            if self.display_grid.canvas.itemcget('eigen', 'state') == tk.HIDDEN:
                state = tk.NORMAL
            else:
                state = tk.HIDDEN
            self.display_grid.canvas.itemconfig('eigen', state=state)

        slider.set(slide_value)
        #slider.set(slide_value) # repeated on purpose but its a bit messy

    def reset(self, data1, data2, slider, gridcontain, eigs):
        slider.set(0)
        data1.set_base()
        data2.set_base()

        data2.bases[0].set(4)
        data2.bases[1].set(1)

        self.display_grid = gridcontain.reset(data1, eigs)
        for co, value in zip(data1.bases, [1, 0, 0, 1]):
            co.set(value)
        for group in eigs:
            for var in group:
                var.set(0)


class MainApplication(tk.Tk):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry('625x800')
        self.title("Transformation Visualiser ")
        visualiser = Visualiser(self)
        visualiser.grid(row=0, column=0, padx=10)


if __name__ == '__main__':
    root = MainApplication()
    root.mainloop()
