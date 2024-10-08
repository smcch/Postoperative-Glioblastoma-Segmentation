#==============================================================================#
#  Author:       * Roberto Romero-Oraá and + Santiago Cepeda                        #
#  Copyright:    * Biomedical Engineering Group                                #
#                + Río Hortega University Hospital                             #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#

import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import threading
import sys
import os
from PIL import Image, ImageTk

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.input_dir = None
        self.output_dir = None
        self.separate_segmentation = tk.BooleanVar(self)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        image = Image.open('logo.png')  # Update with the path to your logo file
        self.logo = ImageTk.PhotoImage(image)
        self.logo_label = tk.Label(self, image=self.logo)
        self.logo_label.pack()

        self.input_dir_label = tk.Label(self, text="Input Directory:")
        self.input_dir_label.pack()

        self.input_dir_button = tk.Button(self)
        self.input_dir_button["text"] = "Choose Directory"
        self.input_dir_button["command"] = self.get_input_directory
        self.input_dir_button.pack()

        self.output_dir_label = tk.Label(self, text="Output Directory:")
        self.output_dir_label.pack()

        self.output_dir_button = tk.Button(self)
        self.output_dir_button["text"] = "Choose Directory"
        self.output_dir_button["command"] = self.get_output_directory
        self.output_dir_button.pack()

        self.separate_segmentation_checkbox = tk.Checkbutton(self)
        self.separate_segmentation_checkbox["text"] = "Separate segmentation layers in different files"
        self.separate_segmentation_checkbox["variable"] = self.separate_segmentation
        self.separate_segmentation_checkbox.pack()

        self.start_button = tk.Button(self)
        self.start_button["text"] = "Start"
        self.start_button["command"] = self.start_preprocess
        self.start_button.pack()

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack()

    def get_input_directory(self):
        self.input_dir = filedialog.askdirectory()
        self.input_dir_label['text'] = f"Input Directory: {self.input_dir}"

    def get_output_directory(self):
        self.output_dir = filedialog.askdirectory()
        self.output_dir_label['text'] = f"Output Directory: {self.output_dir}"

    def start_preprocess(self):
        if self.input_dir is None or self.output_dir is None:
            messagebox.showerror("Error", "Please select both input and output directories.")
            return

        if not os.path.isfile('main.py'):
            messagebox.showerror("Error", "The file main.py could not be found.")
            return

        # Use sys.executable to ensure the same Python interpreter
        command = [sys.executable, 'main.py', '-i', self.input_dir, '-o', self.output_dir]
        if self.separate_segmentation.get():
            command.append('-s')
        thread = threading.Thread(target=self.run_command, args=(command,))
        thread.start()

    def run_command(self, command):
        subprocess.run(command)

root = tk.Tk()
root.title("MRI Preprocessing Pipeline and Glioblastoma Segmentation")
app = Application(master=root)
app.mainloop()
