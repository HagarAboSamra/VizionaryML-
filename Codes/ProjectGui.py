import customtkinter as ctk
from tkinter import ttk, filedialog, messagebox
from PIL import Image
import pandas as pd
from tkinter.constants import CENTER
import os 
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import matplotlib.pyplot as plt

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
        self.app.btn_frame.pack(side='top', fill='x', pady=30)
        self.app.upload_btn = ctk.CTkButton(self.app.btn_frame, text='Upload data', text_color='white', 
                                           fg_color='blue', command=lambda: self.app.logic.switch(self.app.logic.upload), 
                                           state='normal')
        self.app.upload_btn.pack(side='left', padx=15, ipady=2)
        self.app.processing_btn = ctk.CTkButton(self.app.btn_frame, text='Processing', text_color='white', 
                                              fg_color='black', command=lambda: self.app.logic.switch(self.app.logic.processing), 
                                              state='disabled')
        self.app.processing_btn.pack(side='left', padx=15, ipady=2)
        self.app.visualization_btn = ctk.CTkButton(self.app.btn_frame, text='Visualization', text_color='white', 
                                                 fg_color='black', command=lambda: self.app.logic.switch(self.app.logic.visualization), 
                                                 state='disabled')
        self.app.visualization_btn.pack(side='left', padx=15, ipady=2)
        self.app.view_data_btn = ctk.CTkButton(self.app.btn_frame, text='View data', text_color='white', 
                                             fg_color='black', command=lambda: self.app.logic.switch(self.app.logic.view_data), 
                                             state='disabled')
        self.app.view_data_btn.pack(side='left', padx=15, ipady=2)
        self.app.center_frame.pack(side='top', expand=True, fill='both', pady=5, padx=5)
        self.app.ML_modle_btn = ctk.CTkButton(self.app.btn_frame, text='ML modle', text_color='white', 
                                             fg_color='black', command=lambda: self.app.logic.switch(self.app.logic.ML_model), 
                                             state='disabled')
        self.app.ML_modle_btn.pack(side='left', padx=15, ipady=2)
        self.app.center_frame.pack(side='top', expand=True, fill='both', pady=5, padx=5)

# ======================= App Logic and Data Handling =======================
class AppLogic:
    def __init__(self, app):
        self.app = app
        self.app.process= None
        self.processing_flag = False
        self.columns_flag = False
        self.types_refs = {}
        self.btn_refs = {}
        self.selected_plot_button = None
        self.current_plot_type = None
        self.draw_functions = {
            "Pie Chart": self.create_pie_chart,
            "Bar Chart": self.create_bar_chart,
            "Horizontal Bar Chart": self.create_horizontal_bar_chart,
            "Strip Plot": self.create_stripplot,
            "Count Plot": self.create_countplot,
            "Pair Plot": self.create_pairplot,
            "Histogram Plot": self.hist_plot,
            "Box Plot": self.box_plot,
            "Heatmap": self.create_heatmap,
            "KDE Plot": self.create_kde_plot,
            "Scatter Plot": self.create_scatter_plot,
        }
        self.categorical_plots = ["Pie Chart", "Bar Chart", "Horizontal Bar Chart", "Count Plot"]
        self.numerical_plots = ["Pair Plot", "Histogram Plot", "Box Plot", "Heatmap", "KDE Plot", "Scatter Plot"]
        self.two_column_plots = ["Count Plot", "Scatter Plot","Strip Plot"]
        self.both_plots = ["Strip Plot"] 
        self.plot_frame = None
        self.second_column_dropdown = None
        self.second_column_label = None

    def load_image(self, filename):
        return Image.open(os.path.join("images", filename))
      
    def initial_frame(self):
        frame_img = ctk.CTkFrame(self.app.center_frame)
        frame_img.pack()
        try:
            image = self.load_image("image.jpg")
            original_width, original_height = image.size
            image = ctk.CTkImage(image, size=(original_width + 200, original_height + 150))
            img_label = ctk.CTkLabel(frame_img, text=' ', image=image)
            img_label.pack(side='top')
        except FileNotFoundError:
            img_label = ctk.CTkLabel(frame_img, text="Welcome to Data Analysis App", font=("Arial", 24))
            img_label.pack(side='top')

    def switch(self, page):
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
        self.app.process = True
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

    def process_data_type1(self, data):
        pass

    def process_data_type2(self, data):
        pass

    def process_data_type3(self, data):
        pass
    
    def handle_plot_selection(self, name):
        for plot_name, btn in self.btn_refs.items():
            btn.configure(fg_color="#f1f1f1", text_color="black", text=plot_name)

        new_btn = self.btn_refs[name]
        new_btn.configure(fg_color="#147eab", text_color="white", text=f"✓ {name}")
        self.selected_plot_button = new_btn
        self.current_plot_type = name
        self.update_column_dropdowns()

    def update_column_dropdowns(self):
        self.apply_btn.pack(side='left', padx=5, pady=2)
        if not hasattr(self, 'column_dropdown') or not hasattr(self, 'second_column_dropdown'):
            return

        df = self.app.data_processed if self.processing_flag and self.app.data_processed is not None else self.app.data

        if df is None:
            return

        # Hide all column selection widgets by default
        self.column_label.pack_forget()
        self.column_dropdown.pack_forget()
        self.second_column_label.pack_forget()
        self.second_column_dropdown.pack_forget()

        # Special cases for plots that don't need column selection
        if self.current_plot_type in ["Heatmap", "Pair Plot"]:
            return

        # Show first column selector for all other plots
        self.column_label.pack(side='left', padx=5, pady=2)
        self.column_dropdown.pack(side='left', padx=5, pady=2)

        # For Strip Plot: allow all columns in both dropdowns and label their type
        if self.current_plot_type == "Strip Plot":
            cat_cols = self.get_categorical_columns(df)
            num_cols = self.get_numeric_columns(df)
            all_columns = list(df.columns)

            # Build list showing name and type
            labeled_columns = []
            for col in all_columns:
                if col in cat_cols:
                    labeled_columns.append(f"{col}  (categorical)")
                elif col in num_cols:
                    labeled_columns.append(f"{col}  (numerical)")
                else:
                    labeled_columns.append(f"{col}  (other)")

            self.column_label.configure(text="Primary Column:")
            self.column_dropdown.configure(values=labeled_columns)

            self.second_column_label.pack(side='left', padx=5, pady=2)
            self.second_column_dropdown.pack(side='left', padx=5, pady=2)

            self.second_column_label.configure(text="Secondary Column:")
            self.second_column_dropdown.configure(values=labeled_columns)

            # Set default selections
            if labeled_columns:
                self.column_dropdown.set(labeled_columns[0])
                if len(labeled_columns) > 1:
                    self.second_column_dropdown.set(labeled_columns[1])
                else:
                    self.second_column_dropdown.set(labeled_columns[0])
            else:
                self.column_dropdown.set("")
                self.second_column_dropdown.set("")
                messagebox.showerror("Error", "No columns available for Strip Plot.")
            return  # Skip remaining logic

        # For other plots: determine column types
        if self.current_plot_type in self.categorical_plots:
            columns = self.get_categorical_columns(df)
            label_text = "Categorical Column:"
        elif self.current_plot_type in self.numerical_plots:
            columns = self.get_numeric_columns(df)
            label_text = "Numerical Column:"
        else:
            columns = list(df.columns)
            label_text = "Column:"

        self.column_label.configure(text=label_text)
        self.column_dropdown.configure(values=columns)

        if columns:
            self.column_dropdown.set(columns[0])
        else:
            self.column_dropdown.set("")
            messagebox.showerror("Error", f"No suitable columns found for {self.current_plot_type}")
            return

        # Handle other two-column plots (Count Plot, Scatter Plot)
        if self.current_plot_type in self.two_column_plots:
            self.second_column_label.pack(side='left', padx=5, pady=2)
            self.second_column_dropdown.pack(side='left', padx=5, pady=2)

            if self.current_plot_type == "Count Plot":
                second_columns = self.get_categorical_columns(df)
                second_label_text = "Hue Column:"
            elif self.current_plot_type == "Scatter Plot":
                second_columns = self.get_numeric_columns(df)
                second_label_text = "Y-Axis Column:"

            self.second_column_label.configure(text=second_label_text)
            self.second_column_dropdown.configure(values=second_columns)

            if second_columns:
                if len(second_columns) > 1 and self.column_dropdown.get() == second_columns[0]:
                    self.second_column_dropdown.set(second_columns[1])
                else:
                    self.second_column_dropdown.set(second_columns[0])
            else:
                self.second_column_dropdown.set("")
                messagebox.showerror("Error", f"No suitable second column found for {self.current_plot_type}")

    def create_plot_button(self, parent, name):
        btn = ctk.CTkButton(parent, width=175, height=45, text=name,
                            command=lambda n=name: self.handle_plot_selection(n),
                            font=("Arial", 16),
                            fg_color="#f1f1f1", text_color="black")
        btn.pack(pady=5)
        self.btn_refs[name] = btn
        
    def visualization(self):
        self.current_plot_type = None
        right_frame = ctk.CTkFrame(self.app.center_frame, fg_color='transparent')
        right_frame.pack_propagate(False)
        right_frame.pack(padx=5, pady=5, fill='both', expand=True, side='right')

        left_frame = ctk.CTkFrame(self.app.center_frame, width=250, fg_color='#e6dedc', corner_radius=0)
        left_frame.pack(padx=5, pady=5, fill='y', expand=False, side='left')
        left_frame.pack_propagate(False)

        visualization_btns_frame = ctk.CTkFrame(right_frame, fg_color='transparent')
        visualization_btns_frame.pack(side='top', expand=False, fill='x')

        self.before_btn = ctk.CTkButton(visualization_btns_frame, height=40, 
                                      text="Before processing", 
                                      text_color='#A9A9A9', fg_color='#f1f1f1',
                                      command=lambda: self.set_processing_flag(False, self.before_btn, self.after_btn))
        self.before_btn.pack(side='left', padx=5, pady=2)

        self.after_btn = ctk.CTkButton(visualization_btns_frame, height=40, 
                                     text="After processing", 
                                     text_color='#A9A9A9', fg_color='#f1f1f1',
                                     command=lambda: self.set_processing_flag(True, self.after_btn, self.before_btn),
                                     state='disabled' if self.app.process is None else 'normal')
        self.after_btn.pack(side='left', padx=5, pady=2)

        self.apply_btn = ctk.CTkButton(
            visualization_btns_frame,
            height=40,
            text="Apply",
            text_color='white',
            fg_color='#147eab',
            command=self.apply_visualization
        )
        self.apply_btn.pack_forget()

        self.column_label = ctk.CTkLabel(visualization_btns_frame, text="Select Column:", height=40)
        self.column_label.pack_forget()
        
        self.column_dropdown = ctk.CTkComboBox(
            visualization_btns_frame,
            height=40,
            text_color='black',
            fg_color='white',
            border_color='black',
            border_width=1,
            button_color='#147eab',
            dropdown_fg_color='white',
            dropdown_text_color='black',
            dropdown_hover_color='#e6e6e6'
        )
        self.column_dropdown.pack_forget()

        self.second_column_label = ctk.CTkLabel(visualization_btns_frame, text="Second Column:", height=40)
        self.second_column_label.pack_forget()
        
        self.second_column_dropdown = ctk.CTkComboBox(
            visualization_btns_frame,
            height=40,
            text_color='black',
            fg_color='white',
            border_color='black',
            border_width=1,
            button_color='#147eab',
            dropdown_fg_color='white',
            dropdown_text_color='black',
            dropdown_hover_color='#e6e6e6'
        )
        self.second_column_dropdown.pack_forget()

        self.plot_selection_frame = ctk.CTkFrame(left_frame, fg_color='#e6dedc')
        self.plot_selection_frame.pack_forget()

        self.plot_frame = ctk.CTkFrame(right_frame, fg_color='transparent')
        self.plot_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
    def set_processing_flag(self, val, btn_clicked, btn_disabled):
        self.apply_btn.pack_forget()
        self.column_label.pack_forget()
        self.column_dropdown.pack_forget()
        self.second_column_label.pack_forget()
        self.second_column_dropdown.pack_forget()

        """Set processing flag and show plot options"""
        self.processing_flag = val
        
        # Update button states
        if not val:
            btn_clicked.configure(fg_color='black', text='✓ Before processing')
            btn_disabled.configure(fg_color='#f1f1f1', text='After processing')
        else:
            btn_clicked.configure(fg_color='black', text='✓ After processing')
            btn_disabled.configure(fg_color='#f1f1f1', text='Before processing')
        
        # Now show the plot options
        self.show_plot_options()

    def show_plot_options(self):
        """Display plot type selection options with new 'Both' section"""
        for widget in self.plot_selection_frame.winfo_children():
            widget.destroy()
        
        self.plot_selection_frame.pack(fill='both', expand=True)
        
        title = ctk.CTkLabel(self.plot_selection_frame, text='Visualization', 
                            text_color='#147eab', fg_color='transparent', 
                            font=("Arial", 24))
        title.pack(side='top', pady=10)
        
        # Categorical Section
        ctk.CTkLabel(self.plot_selection_frame, text='Categorical', 
                    text_color='#147eab', fg_color='transparent', 
                    font=("Arial", 24)).pack(side='top', pady=10)
        for name in self.categorical_plots:
            self.create_plot_button(self.plot_selection_frame, name) 

        # Numerical Section
        ctk.CTkLabel(self.plot_selection_frame, text='Numerical', 
                    text_color='#147eab', fg_color='transparent', 
                    font=("Arial", 24)).pack(pady=10)
        for name in self.numerical_plots:
            self.create_plot_button(self.plot_selection_frame, name)
            
        # New Both Section
        ctk.CTkLabel(self.plot_selection_frame, text='Both', 
                    text_color='#147eab', fg_color='transparent', 
                    font=("Arial", 24)).pack(pady=10)
        for name in self.both_plots:
            self.create_plot_button(self.plot_selection_frame, name)

    def apply_visualization(self):
        """Create visualization based on selected options"""
        if not self.current_plot_type:
            messagebox.showerror("Error", "Please select a plot type first")
            return

        # Get the appropriate dataset
        if self.processing_flag and self.app.data_processed is not None:
            df = self.app.data_processed  # Use processed data
        else:
            df = self.app.data  # Use original data
        
        if df is None:
            messagebox.showerror("Error", "No data available for visualization")
            return
            
        # Handle special plots
        if self.current_plot_type in ["Heatmap", "Pair Plot"]:
            self.show_plot_in_frame(self.plot_frame, df, None, None, self.current_plot_type)
            return
            
        # Get primary column
        column_name = self.column_dropdown.get()
        if not column_name:
            messagebox.showerror("Error", "Please select a column")
            return
            
        # Clean column name (remove type annotation if present)
        column_name = column_name.split("  (")[0]
        
        # Handle two-column plots
        if self.current_plot_type in self.two_column_plots:
            second_column = self.second_column_dropdown.get()
            if not second_column:
                messagebox.showerror("Error", "Please select a second column")
                return
            second_column = second_column.split("  (")[0]
            self.show_plot_in_frame(self.plot_frame, df, column_name, second_column, self.current_plot_type)
        else:
            # Single column plots
            self.show_plot_in_frame(self.plot_frame, df, column_name, None, self.current_plot_type)

    @staticmethod
    def get_categorical_columns(df):
        return df.select_dtypes(include=["object", "category"]).columns.tolist()

    @staticmethod
    def get_numeric_columns(df):
        return df.select_dtypes(include="number").columns.tolist()

    @staticmethod
    def choose_column(df, column_name, column_type, numeric_required=False):
        if column_name not in df.columns:
            msg = f"Column '{column_name}' not found in the DataFrame."
            print(msg)
            return None

        if column_type == "categorical" and not (
            pd.api.types.is_object_dtype(df[column_name])
            or pd.api.types.is_categorical_dtype(df[column_name])
        ):
            msg = f"Column '{column_name}' is not categorical."
            print(msg)
            return None

        if column_type == "numeric" and not pd.api.types.is_numeric_dtype(df[column_name]):
            msg = f"Column '{column_name}' is not numeric."
            print(msg)
            return None

        if numeric_required and not pd.api.types.is_numeric_dtype(df[column_name]):
            msg = f"Column '{column_name}' must be numeric."
            print(msg)
            return None

        return column_name

    def create_pie_chart(self, ax, df, column_name):
        column = self.choose_column(df, column_name, "categorical")
        if column is None:
            return
        top_categories = df[column].value_counts().head(5)
        ax.pie(
            top_categories.values,
            labels=top_categories.index.tolist(),
            autopct="%1.1f%%",
            radius=1,
            explode=[0.05] * len(top_categories),
            shadow=True,
            textprops={"fontsize": 12},
        )
        ax.set_title(f"Top 5 in '{column}'", fontsize=14)

    def create_bar_chart(self, ax, df, column_name):
        column = self.choose_column(df, column_name, "categorical")
        if column is None:
            return
        value_counts = df[column].value_counts()
        ax.bar(value_counts.index, value_counts.values, color="skyblue", edgecolor="black")
        ax.set_title(f"{column} Frequency")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        ax.tick_params(axis="x", rotation=45)

    def create_horizontal_bar_chart(self, ax, df, column_name):
        column = self.choose_column(df, column_name, "categorical")
        if column is None:
            return
        value_counts = df[column].value_counts().head(10)
        ax.barh(
            value_counts.index, value_counts.values, color="steelblue", edgecolor="black"
        )
        ax.set_title(f"Top 10 {column}")
        ax.set_xlabel("Count")
        ax.set_ylabel(column)

    def create_stripplot(self, ax, df, x_col=None, y_col=None):
        """Flexible stripplot that works with either categorical or numerical columns"""
        # Determine which columns to use
        if x_col is None and y_col is None:
            # Auto-select columns if none provided
            cat_cols = self.get_categorical_columns(df)
            num_cols = self.get_numeric_columns(df)
            
            if cat_cols and num_cols:
                # Default: categorical x vs numerical y
                x_col = cat_cols[0]
                y_col = num_cols[0]
            elif len(num_cols) >= 2:
                # Fallback: numerical vs numerical
                x_col = num_cols[0]
                y_col = num_cols[1]
            else:
                messagebox.showerror("Error", "Need at least one categorical and one numerical column, or two numerical columns")
                return
        
        # Determine plot orientation based on column types
        x_is_cat = x_col in self.get_categorical_columns(df) if x_col else False
        y_is_cat = y_col in self.get_categorical_columns(df) if y_col else False
        
        if x_is_cat and not y_is_cat:
            # Standard case: categorical x vs numerical y
            sns.stripplot(x=x_col, y=y_col, data=df, jitter=True, ax=ax)
            ax.set_title(f"{y_col} distribution by {x_col}")
        elif not x_is_cat and y_is_cat:
            # Flipped case: numerical x vs categorical y
            sns.stripplot(x=x_col, y=y_col, data=df, jitter=True, ax=ax)
            ax.set_title(f"{x_col} distribution by {y_col}")
        elif not x_is_cat and not y_is_cat:
            # Numerical vs numerical
            sns.stripplot(x=x_col, y=y_col, data=df, jitter=True, ax=ax)
            ax.set_title(f"{y_col} vs {x_col}")
        else:
            # Categorical vs categorical (not ideal but possible)
            messagebox.showwarning("Warning", "Strip Plot works best with one categorical and one numerical variable")
            sns.stripplot(x=x_col, y=y_col, data=df, jitter=True, ax=ax)
            ax.set_title(f"{y_col} by {x_col}")

        # Rotate x-axis labels if needed
        if x_is_cat:
            ax.tick_params(axis='x', rotation=45)

    def create_countplot(self, ax, df, primary_col, hue_col):
        if primary_col is None:
            return
        if hue_col is None:
            sns.countplot(
                y=primary_col,
                data=df,
                order=df[primary_col].value_counts().iloc[:10].index,
                ax=ax,
            )
            ax.set_title(f"Top 10 {primary_col}")
        else:
            sns.countplot(
                y=primary_col,
                hue=hue_col,
                data=df,
                order=df[primary_col].value_counts().iloc[:10].index,
                ax=ax,
            )
            ax.set_title(f"Top 10 {primary_col} by {hue_col}")

    def create_pairplot(self, _, df):
        num_cols = self.get_numeric_columns(df)
        hue = self.get_categorical_columns(df)[0] if self.get_categorical_columns(df) else None
        sns.pairplot(df[num_cols + [hue]] if hue else df[num_cols], hue=hue)
        plt.show()

    def hist_plot(self, ax, df, column_name):
        column = self.choose_column(df, column_name, "numeric", numeric_required=True)
        if column is None:
            return
        sns.histplot(df[column], kde=True, color="cornflowerblue", edgecolor="black", ax=ax)
        ax.set_title(f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")

    def box_plot(self, ax, df, column_name):
        column = self.choose_column(df, column_name, "numeric", numeric_required=True)
        if column is None:
            return
        sns.boxplot(x=df[column], color="skyblue", ax=ax)
        ax.set_title(f"Boxplot of {column}")
        ax.set_xlabel(column)

    def create_heatmap(self, ax, df):
        numeric_df = df.select_dtypes(include="number")
        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            linewidths=0.5,
            square=True,
            ax=ax,
        )
        ax.set_title("Correlation Heatmap")

    def create_kde_plot(self, ax, df, column_name):
        column = self.choose_column(df, column_name, "numeric", numeric_required=True)
        if column is None:
            return
        sns.kdeplot(df[column].dropna(), fill=True, color="purple", linewidth=2, ax=ax)
        ax.set_title(f"KDE Plot of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Density")

    def create_scatter_plot(self, ax, df, x_column, y_column):
        if x_column is None or y_column is None:
            return
        sns.scatterplot(data=df, x=x_column, y=y_column, color="teal", ax=ax)
        ax.set_title(f"{y_column} vs {x_column}")
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)

    def show_plot_in_frame(self, frame, df, col1, col2, mode):
        for widget in frame.winfo_children():
            widget.destroy()

        if mode == "Pair Plot":
            self.draw_functions[mode](None, df)
            return

        frame.update_idletasks()
        
        col1 = col1.split("  (")[0] if col1 else None
        col2 = col2.split("  (")[0] if col2 else None

        width_px = frame.winfo_width()
        height_px = frame.winfo_height()
        dpi = 100
        fig = Figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
        ax = fig.add_subplot(111)

        try:
            if mode == "Heatmap":
                self.draw_functions[mode](ax, df)

            elif mode in self.two_column_plots:
                if col1 and col2:
                    self.draw_functions[mode](ax, df, col1, col2)
                else:
                    if mode == "Count Plot":
                        cat_cols = self.get_categorical_columns(df)
                        if len(cat_cols) >= 2:
                            self.draw_functions[mode](ax, df, cat_cols[0], cat_cols[1])
                        else:
                            messagebox.showerror("Error", "Need at least 2 categorical columns for Count Plot")

                    elif mode == "Strip Plot":
                        cat_cols = self.get_categorical_columns(df)
                        num_cols = self.get_numeric_columns(df)
                        if cat_cols and num_cols:
                            self.draw_functions[mode](ax, df, cat_cols[0], num_cols[0])
                        else:
                            messagebox.showerror("Error", "Need at least 1 categorical and 1 numerical column for Strip Plot")

                    elif mode == "Scatter Plot":
                        num_cols = self.get_numeric_columns(df)
                        if len(num_cols) >= 2:
                            self.draw_functions[mode](ax, df, num_cols[0], num_cols[1])
                        else:
                            messagebox.showerror("Error", "Need at least 2 numerical columns for Scatter Plot")

            else:
                # Single-column plots
                if col1:
                    self.draw_functions[mode](ax, df, col1)
                else:
                    if mode in self.categorical_plots:
                        cat_cols = self.get_categorical_columns(df)
                        if cat_cols:
                            self.draw_functions[mode](ax, df, cat_cols[0])
                        else:
                            messagebox.showerror("Error", "No categorical columns found")
                    else:
                        num_cols = self.get_numeric_columns(df)
                        if num_cols:
                            self.draw_functions[mode](ax, df, num_cols[0])
                        else:
                            messagebox.showerror("Error", "No numerical columns found")

            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create plot: {str(e)}")

    def view(self, frame, data_to_show):
        for child in frame.winfo_children():
            child.destroy()
            self.app.root.update()
        tree = ttk.Treeview(frame)
        self.tree_defaults()
        self.view_tables(tree, data_to_show, frame)

    def view_data(self):
        view_data_frame = ctk.CTkFrame(self.app.center_frame, fg_color='transparent')
        button_frame = ctk.CTkFrame(self.app.center_frame, fg_color='transparent')
        button_frame.pack(side='top', fill='x')
        
        try:
            img = self.load_image("image2.jpg")
            image_to_display = ctk.CTkImage(img, size=(800, 600))
            image_label = ctk.CTkLabel(view_data_frame, text=' ', image=image_to_display)
            image_label.pack(side="top", padx=10, pady=10)
        except FileNotFoundError:
            image_label = ctk.CTkLabel(view_data_frame, text="Data Visualization", font=("Arial", 24))
            image_label.pack(side="top", padx=10, pady=10)

        data_before_btn = ctk.CTkButton(button_frame, text='Data before processing', text_color='white', fg_color='black', command=lambda: self.view(view_data_frame, self.app.data))
        data_after_btn = ctk.CTkButton(button_frame, text='Data After processing', text_color='white', fg_color='black', command=lambda: self.view(view_data_frame, self.app.data_processed), state='disabled' if self.app.process is None else 'normal')

        data_before_btn.pack(side='left', padx=10, ipady=2)
        data_after_btn.pack(side='left', padx=15, ipady=2)
        view_data_frame.pack(side='top', expand=True, fill='both', pady=5, padx=5)

    def ML_model(self):
      pass


if __name__ == "__main__":
    DataAnalysisApp()