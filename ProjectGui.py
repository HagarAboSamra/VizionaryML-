import customtkinter as ctk
from tkinter import ttk, filedialog, messagebox
from PIL import Image
import pandas as pd
from tkinter.constants import CENTER
import os 

# ======================= Main Application Class =======================
class DataAnalysisApp:
    def __init__(self):
        self.root = ctk.CTk()
        ctk.set_appearance_mode("light")
        self.root.title('Data Analysis')

        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.root.geometry(f'{self.screen_width}x{self.screen_height}+0+0')

        self.data = None
        self.data_processed = None
        self.process = None
        self.details_window = None
        self.selected_index = None

        self.btn_frame = ctk.CTkFrame(self.root, fg_color='transparent')
        self.center_frame = ctk.CTkFrame(self.root, fg_color='transparent')

        self.ui = AppUI(self)
        self.logic = AppLogic(self)

        self.ui.build_ui()
        self.logic.initial_frame()

        self.root.mainloop()

# ======================= UI Management =======================
class AppUI:
    def __init__(self, app):
        self.app = app

    def build_ui(self):
        # Top button frame for navigation
        self.app.btn_frame.pack(side='top', fill='x', pady=30)

        self.app.upload_btn = ctk.CTkButton(self.app.btn_frame, text='Upload data', text_color='white', fg_color='blue', command=lambda: self.app.logic.switch(self.app.logic.upload), state='normal')
        self.app.upload_btn.pack(side='left', padx=15, ipady=2)

        self.app.processing_btn = ctk.CTkButton(self.app.btn_frame, text='Processing', text_color='white', fg_color='black', command=lambda: self.app.logic.switch(self.app.logic.processing), state='disabled')
        self.app.processing_btn.pack(side='left', padx=15, ipady=2)

        self.app.visualization_btn = ctk.CTkButton(self.app.btn_frame, text='Visualization', text_color='white', fg_color='black', command=lambda: self.app.logic.switch(self.app.logic.visualization), state='disabled')
        self.app.visualization_btn.pack(side='left', padx=15, ipady=2)

        self.app.view_data_btn = ctk.CTkButton(self.app.btn_frame, text='View data', text_color='white', fg_color='black', command=lambda: self.app.logic.switch(self.app.logic.view_data), state='disabled')
        self.app.view_data_btn.pack(side='left', padx=15, ipady=2)

        # Central frame for displaying dynamic content
        self.app.center_frame.pack(side='top', expand=True, fill='both', pady=5, padx=5)

# ======================= App Logic and Data Handling =======================
class AppLogic:
    def __init__(self, app):
        self.app = app
        # To get any image path 
        self.base_dir = os.path.dirname(os.path.abspath(__file__)) 
        self.images_dir = os.path.join(self.base_dir, "images") 
        ''' 
        you have to make path like that ...>
        image_path = os.path.join(self.images_dir, "(name of image).jpg")
        
        '''
    def load_image(self, filename):
        return Image.open(os.path.join("images", filename))
      
    def initial_frame(self):
        # Initial welcome image
        frame_img = ctk.CTkFrame(self.app.center_frame)
        frame_img.pack()
        image = self.load_image("image.jpg")
        original_width, original_height = image.size
        image = ctk.CTkImage(image, size=(original_width + 200, original_height + 150))
        img_label = ctk.CTkLabel(frame_img, text=' ', image=image)
        img_label.pack(side='top')

    def switch(self, page):
        # Clear and show new frame
        for child in self.app.center_frame.winfo_children():
            child.destroy()
            self.app.root.update()
        page()

    def upload(self):
        self.app.process = None
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
        try:
            if file_path:
                if file_path.endswith('.csv'):
                    self.app.data = pd.read_csv(file_path)
                else:
                    self.app.data = pd.read_excel(file_path)

                for btn in self.app.btn_frame.winfo_children():
                    if btn == self.app.upload_btn:
                        btn.configure(text='Replace file')
                    else:
                        btn.configure(state='normal', fg_color='green')

                upload_frame = ctk.CTkFrame(self.app.center_frame, fg_color='transparent')
                upload_frame.pack(side='top', expand=True, fill='both', padx=50)

                tree = ttk.Treeview(upload_frame)
                self.tree_defaults()
                tree.delete(*tree.get_children())
                self.view_tables(tree, self.app.data, upload_frame)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def tree_defaults(self):
        style = ttk.Style()
        style.configure("Treeview.Heading", font=("Arial", 12, "bold"))
        style.configure("Treeview", font=("Arial", 11), rowheight=25)

    def view_tables(self, tree, df, parent_frame):
        try:
            tree['columns'] = ["Index"] + list(df.columns)
            tree['show'] = 'headings'

            tree.column("Index", anchor=CENTER)
            tree.heading("Index", text="Index")

            for col in tree['columns'][1:]:
                tree.column(col, anchor=CENTER)
                tree.heading(col, text=col)

            for idx, row in enumerate(df.to_numpy().tolist()):
                tree.insert('', 'end', values=[df.index[idx]] + row)

            vsb = ttk.Scrollbar(parent_frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=vsb.set)
            vsb.pack(side="right", fill="y")

            hsb = ttk.Scrollbar(parent_frame, orient="horizontal", command=tree.xview)
            tree.configure(xscrollcommand=hsb.set)
            hsb.pack(side="bottom", fill="x")

            tree.pack(pady=20, fill='both', expand=True)
            tree.bind("<ButtonRelease-1>", lambda event: self.on_row_selected(event, tree, df))

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def on_row_selected(self, event, tree, df):
        selected_item = tree.selection()
        if selected_item:
            item_values = tree.item(selected_item[0])["values"]
            self.app.selected_index = item_values[0]
            row_data = df.loc[df.index == self.app.selected_index].iloc[0]
            if self.app.details_window:
                self.app.details_window.destroy()
            self.show_row_details(row_data, df.columns)

    def show_row_details(self, row_data, columns):
        # Display selected row in a popup
        self.app.details_window = ctk.CTkToplevel(self.app.root)
        self.app.details_window.title("Row Details")
        width, height = 500, 400
        x = (self.app.screen_width // 2) - (width // 2)
        y = (self.app.screen_height // 2) - (height // 2)
        self.app.details_window.geometry(f"{width}x{height}+{x}+{y}")
        self.app.details_window.configure(fg_color="white")

        title = ctk.CTkLabel(self.app.details_window, text="Row Details", font=("Arial", 16, "bold"))
        title.pack(pady=10)

        scroll_frame = ctk.CTkScrollableFrame(self.app.details_window, width=460, height=300, fg_color="white")
        scroll_frame.pack(pady=10, padx=20, fill="both", expand=True)

        index_label = ctk.CTkLabel(scroll_frame, text=f"Index: {self.app.selected_index}", anchor="w", font=("Arial", 12), justify="left")
        index_label.pack(anchor="w", pady=5, padx=5)

        for i, value in enumerate(row_data):
            label = ctk.CTkLabel(scroll_frame, text=f"{columns[i]}: {value}", anchor="w", font=("Arial", 12), justify="left")
            label.pack(anchor="w", pady=5, padx=5)

        self.app.view_data_btn.configure(state='disabled')
        self.app.details_window.protocol("WM_DELETE_WINDOW", self.on_close_details)

    def on_close_details(self):
        if self.app.details_window:
            self.app.details_window.destroy()
            self.app.details_window = None
        self.app.view_data_btn.configure(state='normal')

    def processing(self):
        # Placeholder for processing logic
        self.app.process = True

    def visualization(self):
        # Placeholder for visualization logic
        pass

    def view(self, frame, data_to_show):
        # Render DataFrame to Treeview
        for child in frame.winfo_children():
            child.destroy()
            self.app.root.update()
        tree = ttk.Treeview(frame)
        self.tree_defaults()
        self.view_tables(tree, data_to_show, frame)

    def view_data(self):
        # View section for original/processed data
        view_data_frame = ctk.CTkFrame(self.app.center_frame, fg_color='transparent')
        button_frame = ctk.CTkFrame(self.app.center_frame, fg_color='transparent')
        button_frame.pack(side='top', fill='x')
        image2_path = os.path.join(self.images_dir, "image2.jpg")
        img = Image.open(image2_path)
      
        image_to_display = ctk.CTkImage(img, size=(800, 600))
        image_label = ctk.CTkLabel(view_data_frame, text=' ', image=image_to_display)
        image_label.pack(side="top", padx=10, pady=10)

        data_before_btn = ctk.CTkButton(button_frame, text='Data before processing', text_color='white', fg_color='black', command=lambda: self.view(view_data_frame, self.app.data))
        data_after_btn = ctk.CTkButton(button_frame, text='Data After processing', text_color='white', fg_color='black', command=lambda: self.view(view_data_frame, self.app.data_processed), state='disabled' if self.app.process is None else 'normal')

        data_before_btn.pack(side='left', padx=10, ipady=2)
        data_after_btn.pack(side='left', padx=15, ipady=2)
        view_data_frame.pack(side='top', expand=True, fill='both', pady=5, padx=5)

if __name__ == "__main__":
    DataAnalysisApp()
