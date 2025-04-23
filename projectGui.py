import customtkinter as ctk
from tkinter import ttk, filedialog, messagebox
from PIL import Image
import pandas as pd
from tkinter.constants import CENTER

root = ctk.CTk()
ctk.set_appearance_mode("light")
root.title('Data Analysis')
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f'{screen_width}x{screen_height}+{0}+{0}')

def initialFrame():
    frameImg = ctk.CTkFrame(centerFrame)
    frameImg.pack()
    image = Image.open("downloads/main.jpg")
    original_width, original_height = image.size
    image = ctk.CTkImage(image, size=(original_width + 200, original_height + 150))
    img = ctk.CTkLabel(frameImg, text=' ', image=image)
    img.pack(side='top')

def switch(page):
    for child in centerFrame.winfo_children():
        child.destroy()
        root.update()
    page()

def upload():
    global data
    file_path = filedialog.askopenfilename(title='Open file')
    try:
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        else:
            data = pd.read_excel(file_path)
            
        for btns in btnFrame.winfo_children():
            if btns == uploadBtn:
                btns.configure(text='Replace file')
            else:
                btns.configure(state='normal', fg_color='green')

        uploadFrame = ctk.CTkFrame(centerFrame,fg_color='transparent')
        uploadFrame.pack(side='top', expand=True, fill='both',padx=50)
        
        tree = ttk.Treeview(uploadFrame)
        treeDefaults()
        tree.delete(*tree.get_children())
        viewTables(tree, data)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        
def treeDefaults():
    style=ttk.Style()
    style.theme_use("default")
    style.configure("Treeview",background="white",foreground="black",rowheight=25,)    
    style.map("Treeview",background=[('selected', '#0a84ff')])
    
def viewTables(tree, df):
    tree['columns'] = list(df.columns)
    tree['show'] = 'headings'
    for col in tree['columns']:
        tree.column(col, anchor=CENTER)
        tree.heading(col, text=col)
    data_rows=df.to_numpy().tolist()
    for row in data_rows:
        tree.insert('', 'end', values=row)
    tree.pack(pady=20, fill='both', expand=True)

def processing():
    pass
    
def visualization():
    pass

def view(frame, data_to_show):
    for child in frame.winfo_children():
        child.destroy()
        root.update()
    ii = ttk.Treeview(frame)
    treeDefaults()
    viewTables(ii, data_to_show)

def view_data():
    viewDataFrame=ctk.CTkFrame(centerFrame,fg_color='transparent')
    buttonF=ctk.CTkFrame(centerFrame,fg_color='transparent')
    buttonF.pack(side='top', fill='x')
    tree = ttk.Treeview(viewDataFrame)
    treeDefaults()
    dataBefore=ctk.CTkButton(buttonF,text='Data before processing',text_color='white',fg_color='black',command=view(viewDataFrame, data))
    dataAfter=ctk.CTkButton(buttonF,text='Data After processing',text_color='white',fg_color='black',command=lambda: view(viewDataFrame, data))
    dataBefore.pack(side='left',padx=10,ipady=2)
    dataAfter.pack(side='left', padx=15,ipady=2)
    viewDataFrame.pack(side='top', expand=True, fill='both', pady=5,padx=5)
# ===================== Button Frame =====================
btnFrame = ctk.CTkFrame(root, fg_color='transparent')
btnFrame.pack(side='top', fill='x', pady=30)

# Buttons
uploadBtn = ctk.CTkButton(btnFrame,text='Upload data',text_color='white',fg_color='blue',command=lambda:switch(upload),state='normal')
uploadBtn.pack(side='left', padx=15,ipady=2)
processingBtn = ctk.CTkButton(btnFrame,text='Processing',text_color='white',fg_color='black',command=lambda:switch(processing),state='disabled')
processingBtn.pack(side='left', padx=15,ipady=2)
visualizationBtn = ctk.CTkButton(btnFrame,text='Visualization',text_color='white',fg_color='black',command=lambda:switch(visualization),state='disabled')
visualizationBtn.pack(side='left', padx=15,ipady=2)
viewDataBtn = ctk.CTkButton(btnFrame,text='View data',text_color='white',fg_color='black',command=lambda:switch(view_data),state='disable')
viewDataBtn.pack(side='left', padx=15,ipady=2)
# ===================== Center Frame =====================
centerFrame = ctk.CTkFrame(root, fg_color='transparent')
centerFrame.pack(side='top', expand=True, fill='both', pady=5,padx=5)

initialFrame()

root.mainloop()
