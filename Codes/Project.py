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
import tkinter as tk
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
import category_encoders as ce
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def drop_empty_cols(df):
    for i in df:
        if (df[i].isnull().sum()/df.shape[0])/100 >50:
            df.drop(i,inplace=True)

def object_types_to_categorical(df):
    for j in df.select_dtypes(include="object"):
        df[j] = df[j].astype("category")
    return df

class DataAnalysisApp:
    def __init__(self):
        self.root = ctk.CTk()
        ctk.set_appearance_mode("light")
        self.root.title('VizionaryML')
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

class AppUI:
    def __init__(self, app):
        self.app = app

    def build_ui(self):
        self.app.btn_frame.pack(side='top', fill='x', pady=30)
        self.app.upload_btn = ctk.CTkButton(self.app.btn_frame, text='Upload data', text_color='white',
                                            fg_color='#1379ba', command=lambda: self.app.logic.switch(self.app.logic.upload),
                                            state='normal')
        self.app.upload_btn.pack(side='left', padx=150, ipady=2)
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

        self.app.ML_model_btn = ctk.CTkButton(self.app.btn_frame, text='ML Model', text_color='white',
                                                fg_color='black', command=lambda: self.app.logic.switch(self.app.logic.ML_model),
                                                state='disabled')
        self.app.ML_model_btn.pack(side='left', padx=15, ipady=2)
        self.app.center_frame.pack(side='top', expand=True, fill='both', pady=5, padx=5)

class AppLogic:
    def __init__(self, app):
        self.app = app
        self.app.process = None
        self.processing_flag = False
        self.columns_flag = False
        self.types_refs = {}
        self.btn_refs = {}
        self.selected_encode_cols = []
        self.selected_plot_cols = []
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
        self.right_frame = None
        self.cm_image_label = None  # To store the CM image label

        self.initialize_preprocessing_methods()

    def load_image(self, filename):
        return Image.open(os.path.join("images", filename))

    
    def initial_frame(self):
        frame_img = ctk.CTkFrame(self.app.center_frame, fg_color="transparent")
        frame_img.pack()
        
        # Create a modern, colorful label for "VizionaryML"
        title_label = ctk.CTkLabel(
            frame_img,
            text="VizionaryML\n\"Your Data, Your Story\" ",
            font=("Comic sans ms", 35, "bold"),
            text_color="#2596beDEB"  # Vibrant cyan color for modern look
        )
        title_label.pack(side="bottom", pady=10)
        
        try:
            image = self.load_image("image.png")
            original_width, original_height = image.size
            # Resize image to a reasonable size while maintaining aspect ratio
            new_width = 400
            aspect_ratio = original_height / original_width
            new_height = int(new_width * aspect_ratio)
            image = ctk.CTkImage(image, size=(new_width, new_height))
            img_label = ctk.CTkLabel(frame_img, text="", image=image)
            img_label.pack(side='top')
        except FileNotFoundError:
            # Fallback label with same modern style if image fails to load
            img_label = ctk.CTkLabel(
                frame_img,
                text="Welcome to Data Analysis App",
                font=("Roboto", 20),
                text_color="#FF69B4"  # Hot pink for a vibrant, modern contrast
            )
            img_label.pack(side='top', pady=10)

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

                self.app.data_processed = self.app.data.copy()
                self.app.data_processed = object_types_to_categorical(self.app.data_processed)

                for btn in self.app.btn_frame.winfo_children():
                    if btn == self.app.upload_btn:
                        btn.configure(text='Replace file')
                    else:
                        btn.configure(state='normal', fg_color='green')

                upload_frame = ctk.CTkFrame(self.app.center_frame, fg_color='transparent')
                upload_frame.pack(side='top', expand=True, fill='both', padx=50)
                drop_empty_cols(self.app.data_processed)

                tree = ttk.Treeview(upload_frame)
                self.tree_defaults()
                tree.delete(*tree.get_children())
                self.view_tables(tree, self.app.data_processed, upload_frame)

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

    def update_tree(self, df, parent_frame):
        for w in parent_frame.winfo_children():
            w.destroy()
        tree = ttk.Treeview(parent_frame)
        self.tree_defaults()
        tree.delete(*tree.get_children())
        self.view_tables(tree, df, parent_frame)
        
    def simple_imputer(self, df, strategy="mean"):
        if df is None:
            messagebox.showerror("Error", "No data available for imputation")
            return
        try:
            impute=SimpleImputer(strategy=strategy)
            impted=df.select_dtypes(include='number').columns.tolist()
            df[impted]=impute.fit_transform(df[impted])
            messagebox.showinfo("Success", f"Applied {strategy} imputation successfully")      
        except Exception as e:
            messagebox.showerror("Error", f"Failed to impute missing values: {str(e)}")
        
    def fill_categorical(self, df, strategy="most_frequent", fill_value="None"):
        if df is None:
            return
        have_null = df.columns[df.isna().any()]
        categorical = df[have_null].select_dtypes(include="category")
        if categorical.empty:
            messagebox.showinfo("Info", "No categorical columns with missing values found")
            return
        
        try:
            if strategy == "most_frequent":
                imp = SimpleImputer(strategy=strategy)
            else:
                imp = SimpleImputer(strategy='constant' ,fill_value='NONE')
            df[categorical.columns] = imp.fit_transform(categorical)    
            messagebox.showinfo("Success", f"Applied {strategy} imputation to categorical columns")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to impute missing values: {str(e)}")

    def k_mean_imputer(self, df, n_value=5):
        if df is None:
            return
        cols_null = df.columns[df.isna().any()]
        cols_num = df[cols_null].select_dtypes(include=[np.number])
        if cols_num.empty:
            messagebox.showinfo("Info", "No numeric columns with missing values found")
            return
        
        try:
            imputer = KNNImputer(n_neighbors=n_value)
            df[cols_num.columns] = imputer.fit_transform(cols_num)
            messagebox.showinfo("Success", f"Applied KNN imputation with {n_value} neighbors")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to impute missing values: {str(e)}")

    def iterative_imputer(self, df):
        if df is None:
            return
        cols_null = df.columns[df.isna().any()]
        cols_num = df[cols_null].select_dtypes(include=[np.number]).columns
        if len(cols_num) == 0:
            messagebox.showinfo("Info", "No numeric columns with missing values found")
            return
        
        try:
            imp = IterativeImputer(max_iter=10, random_state=0)
            df[cols_num] = imp.fit_transform(df[cols_num])
            messagebox.showinfo("Success", "Applied iterative imputation to numeric columns")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to impute missing values: {str(e)}")

    def outliers_z(self, df):
        if df is None:
            return
        num = df.select_dtypes(include=[np.number])
        if num.empty:
            messagebox.showinfo("Info", "No numeric columns found for outlier detection")
            return
        
        try:
            z = np.abs(stats.zscore(num))
            out = z[(z > 3).any(axis=1)]
            df.drop(out.index, inplace=True)
            messagebox.showinfo("Success", f"Removed {len(out)} outliers using Z-score method")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove outliers: {str(e)}")

    def outliers_iqr(self, df):
        if df is None:
            return
        num = df.select_dtypes(include=[np.number])
        if num.empty:
            messagebox.showinfo("Info", "No numeric columns found for outlier detection")
            return
        
        try:
            q1, q3 = num.quantile(0.25), num.quantile(0.75)
            iqr = q3 - q1
            mask = (num < (q1 - 1.5 * iqr)) | (num > (q3 + 1.5 * iqr))
            outliers = mask[mask.any(axis=1)]
            df.drop(outliers.index, inplace=True)
            messagebox.showinfo("Success", f"Removed {len(outliers)} outliers using IQR method")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove outliers: {str(e)}")

    def label_encoder(self, df, cols):
        try:
            enc = LabelEncoder()
            for c in cols:
                df[c] = enc.fit_transform(df[c])
            messagebox.showinfo("Success", f"Applied Label Encoding to {len(cols)} columns")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply Label Encoding: {str(e)}")

    def one_hot(self, df, cols):
        try:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            enc_arr = enc.fit_transform(df[cols])
            enc_df = pd.DataFrame(enc_arr, columns=enc.get_feature_names_out(cols))
            df.drop(cols, axis=1, inplace=True)
            df.reset_index(drop=True, inplace=True)
            df[enc_df.columns] = enc_df
            messagebox.showinfo("Success", f"Applied One-Hot Encoding to {len(cols)} columns")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply One-Hot Encoding: {str(e)}")

    def binary_enc(self, df, cols):
        try:
            enc = ce.BinaryEncoder(cols=cols)
            new = enc.fit_transform(df[cols])
            df.drop(cols, axis=1, inplace=True)
            df[new.columns] = new
            messagebox.showinfo("Success", f"Applied Binary Encoding to {len(cols)} columns")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply Binary Encoding: {str(e)}")

    def ordinal_enc(self, df, cols):
        try:
            enc = ce.OrdinalEncoder(cols=cols)
            df[cols] = enc.fit_transform(df[cols])
            messagebox.showinfo("Success", f"Applied Ordinal Encoding to {len(cols)} columns")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply Ordinal Encoding: {str(e)}")

    def minmax_scaler(self, df):
        if df is None:
            return
        try:
            scaler = MinMaxScaler()
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            messagebox.showinfo("Success", "Applied Min-Max scaling to numeric columns")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply Min-Max scaling: {str(e)}")

    def standard_scaler(self, df):
        if df is None:
            return
        try:
            scaler = StandardScaler()
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            messagebox.showinfo("Success", "Applied Standard scaling to numeric columns")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply Standard scaling: {str(e)}")

    def handle_duplicates(self, df):
        if df is None:
            return
        try:
            before = len(df)
            df.drop_duplicates(inplace=True)
            after = len(df)
            removed = before - after
            messagebox.showinfo("Success", f"Removed {removed} duplicate rows")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove duplicates: {str(e)}")

    def initialize_preprocessing_methods(self):
        self.numerical_missing = [
            ("Simple imputer mean", lambda: self.simple_imputer(self.app.data_processed, "mean")),
            ("Simple imputer median", lambda: self.simple_imputer(self.app.data_processed, "median")),
            ("Simple imputer mode", lambda: self.simple_imputer(self.app.data_processed, "most_frequent")),
            ("KNN imputer", lambda: self.k_mean_imputer(self.app.data_processed)),
            ("Iterative imputer", lambda: self.iterative_imputer(self.app.data_processed)),
        ]
        self.categorical_missing = [
            ("Simple imputer mode", lambda: self.fill_categorical(self.app.data_processed, "most_frequent")),
            ("Simple imputer constant", lambda: self.fill_categorical(self.app.data_processed, "constant")),
        ]
        self.duplicates = [("Handling duplicates", lambda: self.handle_duplicates(self.app.data_processed))]
        self.outliers = [
            ("Z score", lambda: self.outliers_z(self.app.data_processed)),
            ("IQR method", lambda: self.outliers_iqr(self.app.data_processed))
        ]
        self.encoding_methods = [
            ("One hot encoder", lambda: None),
            ("Label encoder", lambda: None),
            ("Binary Encoder", lambda: None),
            ("Ordinal encoder", lambda: None),
        ]
        self.normalization = [
            ("Mini-max scaler", lambda: self.minmax_scaler(self.app.data_processed)),
            ("Standard scaler", lambda: self.standard_scaler(self.app.data_processed))
        ]
        self.skew = [
            ("Data Skewness",lambda: None)
        ]
        self.combo_defs = [
            ("numerical NaN", self.numerical_missing),
            ("categorical NaN", self.categorical_missing),
            ("duplicates", self.duplicates),
            ("outliers", self.outliers),
            ("encoding", self.encoding_methods),
            ("normalization", self.normalization),
            ("skewness handeling",self.skew)
        ]

    def execute(self, choice):
        for title, lst in self.combo_defs:
            for name, func in lst:
                if name == choice:
                    func()
                    self.update_tree(self.app.data_processed, self.right_frame)
                    return

    def processing(self):
        self.app.process = True
        font = ("Arial", 16)
    
        for widget in self.app.center_frame.winfo_children():
            widget.destroy()
    
        left_frame = ctk.CTkFrame(self.app.center_frame, width=250, fg_color='#e6dedc', corner_radius=0)
        left_frame.pack(side="left", fill="y", padx=15, pady=10)
        left_frame.pack_propagate(False)
    
        self.right_frame = ctk.CTkFrame(self.app.center_frame)
        self.right_frame.pack(side="right", expand=True, fill="both", padx=5, pady=15)
    
        title = ctk.CTkLabel(left_frame, text='Processing', text_color='#147eab', 
                            fg_color='transparent', font=("Arial", 24))
        title.pack(side='top', pady=20)
    
        for title_group, lst in self.combo_defs:
            if title_group == "encoding":
                self.encoding_cb = ctk.CTkComboBox(left_frame, values=[n for n, _ in lst], 
                                                 width=180, command=self.update_encoding_checklist)
                self.encoding_cb.set(title_group)
                self.encoding_cb.pack(pady=6)
            elif title_group == "skewness handeling":
                self.skew_cb = ctk.CTkComboBox(left_frame, values=[n for n, _ in lst], 
                                                 width=180, command=self.update_encoding_checklist)
                self.skew_cb.pack(pady=6)
                self.skew_cb.set(title_group)
            else:
                cb = ctk.CTkComboBox(left_frame, values=[n for n, _ in lst], 
                                    width=180, command=self.execute)
                cb.set(title_group)
                cb.pack(pady=6)
    
        self.encode_frame = ctk.CTkScrollableFrame(left_frame, width=190, height=140)
        self.encode_frame.pack(fill="both", expand=True, pady=6)
    
        self.encode_apply_btn = ctk.CTkButton(left_frame, text="Apply Encoding", width=180,
                                            command=self.apply_selected_encoding)
        self.encode_apply_btn.pack(pady=5)

        self.skew_apply_btn = ctk.CTkButton(left_frame, text="Handle skewness", width=180,
                                            command=self.apply_log_transform)
        self.skew_apply_btn.pack(pady=10)
    
        self.proc_tree = ttk.Treeview(self.right_frame)
        self.tree_defaults()
        self.build_encode_checklist(self.app.data_processed, self.encode_frame)
        self.view_tables(self.proc_tree, self.app.data_processed, self.right_frame)
    
    def update_encoding_checklist(self, choice):
        if choice in ["One hot encoder", "Label encoder", "Binary Encoder", "Ordinal encoder"]:
            if self.app.data_processed is not None:
                self.build_encode_checklist(self.app.data_processed, self.encode_frame)
        elif choice in "Data Skewness":
            self.handle_skewness(self.app.data_processed, self.encode_frame)
        
    def build_encode_checklist(self, df, parent_frame):
        for w in parent_frame.winfo_children():
            w.destroy() 
        if df is None:
            return
        
        categorical_cols = df.select_dtypes(include="category").columns
        if len(categorical_cols) == 0:
            label = ctk.CTkLabel(parent_frame, text="No categorical columns found")
            label.pack()
            return
        
        for col in categorical_cols:
            var = tk.BooleanVar(value=False)
            
            def toggle_column(column=col, variable=var):
                if variable.get():
                    if column not in self.selected_encode_cols:
                        self.selected_encode_cols.append(column)
                else:
                    if column in self.selected_encode_cols:
                        self.selected_encode_cols.remove(column)
            
            cb = ctk.CTkCheckBox(parent_frame, text=col, variable=var, 
                                command=lambda c=col, v=var: toggle_column(c, v))
            cb.pack(anchor="w")

    def handle_skewness(self, df, left_frame, skew_threshold=1.0):
        for widget in left_frame.winfo_children():
            widget.destroy()
        if df is None:
            return
    
        numeric_cols = df.select_dtypes(include='number').columns
        skewness = df[numeric_cols].apply(lambda x: x.skew()).abs()
        skewed_cols = skewness[skewness > skew_threshold]
    
        if skewed_cols.empty:
            ctk.CTkLabel(left_frame, text="No highly skewed numeric columns found.").pack(pady=10)
            return
    
        self.log_var_dict = {}
    
        for col in skewed_cols.index:
            var = ctk.BooleanVar()
            cb = ctk.CTkCheckBox(left_frame, text=f"{col} (skew: {df[col].skew():.2f})", variable=var)
            cb.pack(anchor="w", padx=20)
            self.log_var_dict[col] = var

    def apply_log_transform(self):
        selected_cols = [col for col, var in self.log_var_dict.items() if var.get()]
        if not selected_cols:
            return

        for col in selected_cols:
            if (self.app.data_processed[col] < 0).any():
                print(f"Skipping '{col}' - contains negative values.")
                continue
            self.app.data_processed[col] = np.log1p(self.app.data_processed[col])

        self.update_tree(self.app.data_processed, self.right_frame)
    
    def apply_selected_encoding(self):
        if not self.selected_encode_cols or self.app.data_processed is None:
            messagebox.showwarning("Warning", "No columns selected for encoding")
            return
            
        encoding_method = self.encoding_cb.get()
        try:
            if encoding_method == "One hot encoder":
                self.one_hot(self.app.data_processed, self.selected_encode_cols)
            elif encoding_method == "Label encoder":
                self.label_encoder(self.app.data_processed, self.selected_encode_cols)
            elif encoding_method == "Binary Encoder":
                self.binary_enc(self.app.data_processed, self.selected_encode_cols)
            elif encoding_method == "Ordinal encoder":
                self.ordinal_enc(self.app.data_processed, self.selected_encode_cols)
                
            self.update_tree(self.app.data_processed, self.right_frame)
            self.build_encode_checklist(self.app.data_processed, self.encode_frame)
            self.selected_encode_cols = []
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during encoding: {str(e)}")

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
            annot=False,
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

    def show_cm_in_frame(self, frame, cm_list, model_name, class_labels):
        """Save and display multiple confusion matrices in the given frame"""
        # Clear previous content in the frame
        for widget in frame.winfo_children():
            widget.destroy()

        try:
            # Ensure the 'images' directory exists
            os.makedirs("images", exist_ok=True)

            # Create and save each confusion matrix plot
            for cm_type, cm in cm_list:
                fig = plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=class_labels, yticklabels=class_labels)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title(f'Confusion Matrix - {model_name} ({cm_type})')
                plt.tight_layout()

                # Save the plot as a PNG file
                cm_image_path = os.path.join("images", f"cm_{cm_type.lower().replace(' ', '_')}_{model_name.lower().replace(' ', '_')}.png")
                plt.savefig(cm_image_path, dpi=100, bbox_inches='tight')
                plt.close(fig)

                # Load and display the image in the GUI
                image = Image.open(cm_image_path)
                image = ctk.CTkImage(image, size=(400, 350))  # Adjust size to fit UI
                label = ctk.CTkLabel(frame, text='', image=image)
                label.pack(fill='x', expand=True, padx=10, pady=5)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to display confusion matrix: {str(e)}")

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
        frame = ctk.CTkFrame(self.app.center_frame)
        frame.pack(expand=True, fill="both", padx=20, pady=20)

        # Define all variables as instance attributes
        self.selected_model = None
        self.feature_columns = []
        self.target_column = None
        self.X, self.y = None, None
        self.cm_frame = None  # Frame for displaying the CM

        def select_model(name):
            models_map = {
                "Decision Tree": DecisionTreeClassifier(),
                "KNN": KNeighborsClassifier(),
                "SVM": SVC(),
                "Naive Bayes": GaussianNB(),
                "Random Forest": RandomForestClassifier(),
                "KMeans": KMeans(n_clusters=3)
            }
            self.selected_model = models_map.get(name)
            result_box.delete("0.0", "end")
            result_box.insert("0.0", f"Selected Model: {name}")

        def update_features():
            if self.app.data is not None:
                feature_listbox.delete(0, "end")
                for col in self.app.data.columns:
                    feature_listbox.insert("end", col)

                target_menu.configure(values=list(self.app.data.columns))
                target_menu.set(self.app.data.columns[0])

        def set_features():
            selected = feature_listbox.curselection()
            self.feature_columns = [feature_listbox.get(i) for i in selected]
            result_box.insert("end", f"\nSelected Features: {self.feature_columns}")

        def set_target():
            self.target_column = target_menu.get()
            result_box.insert("end", f"\nSelected Target: {self.target_column}")

        def prepare_data():
            if self.app.data is not None and self.feature_columns and self.target_column:
                self.X = self.app.data[self.feature_columns]
                self.y = self.app.data[self.target_column]
                result_box.insert("end", "\nData is ready for training!")
                return True
            else:
                result_box.insert("end", "\nPlease select features and target first!")
                return False

        def evaluate_model():
            if self.selected_model is None:
                result_box.insert("end", "\nPlease select a model first!")
                return
            if not prepare_data():
                return

            # Get unique class labels
            class_labels = np.sort(self.y.unique()).astype(str).tolist()

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            self.selected_model.fit(X_train, y_train)
            y_pred = self.selected_model.predict(X_test)

            cm_split = confusion_matrix(y_test, y_pred)
            acc_split = accuracy_score(y_test, y_pred)
            prec_split = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec_split = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1_split = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # K-Fold Cross-Validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_metrics = []
            cm_kfold_total = None  # To aggregate K-Fold CMs
            for fold, (train_idx, test_idx) in enumerate(kf.split(self.X), 1):
                X_train_fold, X_test_fold = self.X.iloc[train_idx], self.X.iloc[test_idx]
                y_train_fold, y_test_fold = self.y.iloc[train_idx], self.y.iloc[test_idx]
                self.selected_model.fit(X_train_fold, y_train_fold)
                y_pred_fold = self.selected_model.predict(X_test_fold)

                cm_fold = confusion_matrix(y_test_fold, y_pred_fold)
                acc_fold = accuracy_score(y_test_fold, y_pred_fold)
                prec_fold = precision_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0)
                rec_fold = recall_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0)
                f1_fold = f1_score(y_test_fold, y_pred_fold, average='weighted', zero_division=0)

                fold_metrics.append({
                    'fold': fold,
                    'cm': cm_fold,
                    'acc': acc_fold,
                    'prec': prec_fold,
                    'rec': rec_fold,
                    'f1': f1_fold
                })

                # Aggregate K-Fold CM
                if cm_kfold_total is None:
                    cm_kfold_total = cm_fold
                else:
                    cm_kfold_total += cm_fold

            # Compute average metrics for K-Fold
            acc_kf_avg = np.mean([m['acc'] for m in fold_metrics])
            prec_kf_avg = np.mean([m['prec'] for m in fold_metrics])
            rec_kf_avg = np.mean([m['rec'] for m in fold_metrics])
            f1_kf_avg = np.mean([m['f1'] for m in fold_metrics])

            # Function to format confusion matrix as text
            def format_cm(cm):
                lines = ["Confusion Matrix:"]
                for row in cm:
                    lines.append("[" + ", ".join(f"{val:>4}" for val in row) + "]")
                return "\n".join(lines)

            # Prepare the result text
            result = "Train-Test Split:\n"
            result += format_cm(cm_split) + "\n"
            result += f"Accuracy: {acc_split:.4f}\n"
            result += f"Precision: {prec_split:.4f}\n"
            result += f"Recall: {rec_split:.4f}\n"
            result += f"F1-Score: {f1_split:.4f}\n\n"

            result += "K-Fold Cross-Validation (5 folds):\n"
            result += "Aggregated Confusion Matrix:\n"
            result += format_cm(cm_kfold_total) + "\n"
            result += "Average Metrics:\n"
            result += f"Accuracy: {acc_kf_avg:.4f}\n"
            result += f"Precision: {prec_kf_avg:.4f}\n"
            result += f"Recall: {rec_kf_avg:.4f}\n"
            result += f"F1-Score: {f1_kf_avg:.4f}\n\n"

            result += "Details per fold:\n"
            for fold in fold_metrics:
                result += f"Fold {fold['fold']}:\n"
                result += format_cm(fold['cm']) + "\n"
                result += f"Accuracy: {fold['acc']:.4f}\n"
                result += f"Precision: {fold['prec']:.4f}\n"
                result += f"Recall: {fold['rec']:.4f}\n"
                result += f"F1-Score: {fold['f1']:.4f}\n\n"

            # Insert the result into the text box
            result_box.delete("0.0", "end")
            result_box.insert("0.0", result)

            # Display the confusion matrix plots in the GUI
            model_name = [name for name, model in [
                ("Decision Tree", DecisionTreeClassifier()),
                ("KNN", KNeighborsClassifier()),
                ("SVM", SVC()),
                ("Naive Bayes", GaussianNB()),
                ("Random Forest", RandomForestClassifier()),
                ("KMeans", KMeans(n_clusters=3))
            ] if model.__class__ == self.selected_model.__class__][0]
            cm_list = [
                ("Train-Test Split", cm_split),
                ("K-Fold Aggregated", cm_kfold_total)
            ]
            self.show_cm_in_frame(self.cm_frame, cm_list, model_name, class_labels)

        # ============ UI Layout ============

        top_frame = ctk.CTkFrame(frame)
        top_frame.pack(fill="x", pady=50)

        for model in ["Decision Tree", "KNN", "SVM", "Naive Bayes", "Random Forest", "KMeans"]:
            ctk.CTkButton(top_frame, text=model, width=100, command=lambda m=model: select_model(m)).pack(side="left", padx=5)

        body_frame = ctk.CTkFrame(frame)
        body_frame.pack(expand=True, fill="both", pady=10)

        side_frame = ctk.CTkFrame(body_frame, width=300)
        side_frame.pack(side="left", fill="y", padx=10)

        ctk.CTkLabel(side_frame, text="Select Features").pack(pady=5)
        feature_listbox = tk.Listbox(side_frame, selectmode="multiple", height=10)
        feature_listbox.pack(pady=5)

        ctk.CTkButton(side_frame, text="Set Features", command=set_features).pack(pady=5)

        ctk.CTkLabel(side_frame, text="Target Column").pack(pady=5)
        target_menu = ctk.CTkComboBox(side_frame, width=180)
        target_menu.pack(pady=5)

        ctk.CTkButton(side_frame, text="Set Target", command=set_target).pack(pady=5)
        ctk.CTkButton(side_frame, text="Evaluate Model", command=evaluate_model).pack(pady=10)

        # Create a frame for the results and CM plot
        result_frame = ctk.CTkFrame(body_frame)
        result_frame.pack(side="right", fill="both", expand=True, padx=20)

        # Text box for metrics
        result_box = ctk.CTkTextbox(result_frame, width=300)
        result_box.pack(side="top", fill="both", expand=True, padx=10, pady=5)
        result_box.insert("0.0", "🔍 Model results will appear here...")

        # Scrollable frame for the confusion matrix plots
        self.cm_frame = ctk.CTkScrollableFrame(result_frame, height=400)
        self.cm_frame.pack(side="top", fill="both", expand=True, padx=10, pady=5)

        update_features()

if __name__ == "__main__":
    DataAnalysisApp()