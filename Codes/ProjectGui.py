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
        
        # Flags and state
        self.process = False
        self.processing_flag = False
        self.columns_flag = False
        self.types_refs = {}
        self.btn_refs = {}

    # To get any image path 
    def load_image(self, filename):
        return Image.open(os.path.join("images", filename))
    ''' 
    you have to load image use ...>
    image = self.load_image("image.jpg")
        
    '''
      
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

    # Function to start data processing
    def processing(self):
        self.process = True
        font = ("Arial", 16)

        right_frame = ctk.CTkFrame(self.app.center_frame)
        right_frame.pack(fill='both', side='right', expand=True, padx=5, pady=15)
        left_frame = ctk.CTkFrame(self.app.center_frame, width=250, fg_color='#e6dedc', corner_radius=0)
        left_frame.pack(fill='y', side='left', padx=15, pady=10, ipadx=10, ipady=10)

        left_frame.pack_propagate(False)
        right_frame.pack_propagate(False)

        title = ctk.CTkLabel(left_frame, text='Processing', text_color='#147eab', fg_color='transparent', font=("Arial", 24))
        title.pack(side='top', pady=30)

        types = [
            ("Type 1", lambda: self.process_data_type1(self.app.data)),
            ("Type 2", lambda: self.process_data_type2(self.app.data)),
            ("Type 3", lambda: self.process_data_type3(self.app.data))
        ]
        for txt, command in types:
            btn = ctk.CTkButton(left_frame, width=175, height=50, text=txt, font=font, command=command)
            btn.pack(side='top', expand=False, pady=10)
            self.types_refs[txt] = btn

    #function to assign dataProcess according to the chosen type
    def process_data_type1(self, data):
        # view(rightFrame, dataProcessed)
        pass

    def process_data_type2(self, data):
        pass

    def process_data_type3(self, data):
        pass
    
    # Placeholder function for visualization (currently empty)
    def visualization(self):
        right_frame = ctk.CTkFrame(self.app.center_frame, fg_color='transparent')
        right_frame.pack_propagate(False)
        right_frame.pack(padx=5, pady=5, fill='both', expand=True, side='right')

        left_frame = ctk.CTkFrame(self.app.center_frame, width=250, fg_color='#e6dedc', corner_radius=0)
        left_frame.pack(padx=5, pady=5, fill='y', expand=False, side='left')
        left_frame.pack_propagate(False)

        visualization_btns_frame = ctk.CTkFrame(right_frame, fg_color='transparent')
        visualization_btns_frame.pack(side='top', expand=False, fill='x')

        before_btn = ctk.CTkButton(visualization_btns_frame, height=40, text="Before processing", text_color='#A9A9A9', fg_color='#f1f1f1',
                                   command=lambda: self.set_processing_flag(False, before_btn, after_btn))
        before_btn.pack(side='left', padx=5, pady=2)

        after_btn = ctk.CTkButton(visualization_btns_frame, height=40, text="After processing", text_color='#A9A9A9', fg_color='#f1f1f1',
                                  command=lambda: self.set_processing_flag(True, after_btn, before_btn))
        after_btn.pack(side='left', padx=5, pady=2)

        all_btn = ctk.CTkButton(visualization_btns_frame, height=40, text="All data", text_color='#A9A9A9', fg_color='#f1f1f1',
                                command=lambda: self.set_columns_flag(False, all_btn, specific_btn))
        all_btn.pack(side='left', padx=5, pady=2)

        specific_btn = ctk.CTkButton(visualization_btns_frame, height=40, text="Specific columns", text_color='#A9A9A9', fg_color='#f1f1f1',
                                     command=lambda: self.set_columns_flag(True, specific_btn, all_btn))
        specific_btn.pack(side='left', padx=5, pady=2)
        self.label_entry = ctk.CTkEntry(
            visualization_btns_frame,
            height=40,
            text_color='black',
            fg_color='white',
            border_color='black',
            border_width=1,
            placeholder_text="Column Name"
        )
        self.label_entry.pack_forget()

        
        title = ctk.CTkLabel(left_frame, text='Visualization', text_color='#147eab', fg_color='transparent', font=("Arial", 24))
        title.pack(side='top', pady=30)

        title = ctk.CTkLabel(left_frame, text='Category', text_color='#147eab', fg_color='transparent', font=("Arial", 24))
        title.pack(side='top', pady=30)
        
        button_data = [
            ("Type 1", lambda: self.visualize_data_type1(self.app.data)),
            ("Type 2", lambda: self.visualize_data_type2(self.app.data)),
            ("Type 3", lambda: self.visualize_data_type3(self.app.data))
        ]
        for name, command in button_data:
            btn = ctk.CTkButton(left_frame, width=175, height=50, text=name, command=command, font=("Arial", 16))
            btn.pack(side='top', expand=False, pady=10)
            self.btn_refs[name] = btn #reference to reach buttons referring to visualization_data_types
        title = ctk.CTkLabel(left_frame, text='numeric', text_color='#147eab', fg_color='transparent', font=("Arial", 24))
        title.pack(side='top', pady=30)

    
    def visualize_data_type1(self, data):
        pass

    def visualize_data_type2(self, data):
        pass

    def visualize_data_type3(self, data):
        pass

    #function to determine whether to represent data before or after processing
    def set_processing_flag(self, val, btn_clicked, btn_disabled):
        #not processsing_flag means data before processing else it's data after processing
        self.processing_flag = val
        if not val:
            btn_clicked.configure(fg_color='black', text='✓ Before processing')
            btn_disabled.configure(fg_color='#f1f1f1', text='After processing')
        else:
            btn_clicked.configure(fg_color='black', text='✓ After processing')
            btn_disabled.configure(fg_color='#f1f1f1', text='Before processing')

    def set_columns_flag(self, val, btn_clicked, btn_disabled):
        self.columns_flag = val
        if not val:
            btn_clicked.configure(fg_color='black', text='✓ All data')
            btn_disabled.configure(fg_color='#f1f1f1', text='Specific columns')
            self.label_entry.pack_forget()
        else:
            btn_clicked.configure(fg_color='black', text='✓ Specific columns')
            btn_disabled.configure(fg_color='#f1f1f1', text='All data')
            self.label_entry.pack(side='left', padx=5, pady=2)

    #function to apply selected visualization features(before,after,all data,....)
    def apply(self):
        df = self.app.data if not self.processing_flag else self.app.data_processed
        print("Using processed data" if self.processing_flag else "Using original data")
        print("Showing specific columns" if self.columns_flag else "Showing all columns")

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
        img = self.load_image("image2.jpg")
      
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
