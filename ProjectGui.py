import customtkinter as ctk
from tkinter import ttk, filedialog, messagebox
from PIL import Image
import pandas as pd
from tkinter.constants import CENTER

process = None  # Flag to track processing status

root = ctk.CTk()  # Main window
ctk.set_appearance_mode("light")  # Set light theme
root.title('Data Analysis')  # Window title
screen_width = root.winfo_screenwidth()  # Get screen width
screen_height = root.winfo_screenheight()  # Get screen height
root.geometry(f'{screen_width}x{screen_height}+{0}+{0}')  # Set window size and position

# Variables to hold data before and after processing
details_window = None  # Store details window reference
selected_index = None  # Store index of selected row
data = None  # Original data
dataProcessed = None  # Processed data

# Function to display initial content (an image)
def initialFrame():
    frameImg = ctk.CTkFrame(centerFrame)  # Frame to hold the image
    frameImg.pack()
    image = Image.open("image.jpg")  # Image path (static)
    original_width, original_height = image.size  # Get original image size
    image = ctk.CTkImage(image, size=(original_width + 200, original_height + 150))  # Resize image
    img = ctk.CTkLabel(frameImg, text=' ', image=image)  # Label to hold the image
    img.pack(side='top')  # Pack label into the frame

# Function to switch between pages (frames)
def switch(page):
    for child in centerFrame.winfo_children():  # Clear current page content
        child.destroy()
        root.update()
    page()  # Call the page function to load new content

# Function to upload data (CSV or Excel files)
def upload():
    global data,process
    process=None
    file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]  # File types to choose from
        )
    try:
        if file_path:
            if file_path.endswith('.csv'):  # If the file is CSV
                data = pd.read_csv(file_path)
            else:  # If the file is Excel
                data = pd.read_excel(file_path)
            
            # Update button text and state
            for btns in btnFrame.winfo_children():
                if btns == uploadBtn:
                    btns.configure(text='Replace file')
                else:
                    btns.configure(state='normal', fg_color='green')

            uploadFrame = ctk.CTkFrame(centerFrame, fg_color='transparent')  # Frame to display data
            uploadFrame.pack(side='top', expand=True, fill='both', padx=50)
            
            tree = ttk.Treeview(uploadFrame)  # Treeview widget to display data
            treeDefaults()  # Set default style for treeview
            tree.delete(*tree.get_children())  # Clear previous data
            viewTables(tree, data, uploadFrame)  # Populate treeview with data

    except Exception as e:  # Handle errors during upload
        messagebox.showerror("Error", f"An error occurred: {e}")

# Set style for Treeview widget
def treeDefaults():
    style = ttk.Style()
    style.configure("Treeview.Heading", font=("Arial", 12, "bold"))
    style.configure("Treeview", font=("Arial", 11), rowheight=25)

# Function to display data in Treeview
def viewTables(tree, df, parent_frame):
    try:
        tree['columns'] = ["Index"] + list(df.columns)  # Set column headers
        tree['show'] = 'headings'
        
        tree.column("Index", anchor=CENTER)  # Center-align Index column
        tree.heading("Index", text="Index")
        
        # Create columns for each data field
        for col in tree['columns'][1:]:
            tree.column(col, anchor=CENTER)
            tree.heading(col, text=col)

        # Insert rows into the Treeview
        data_rows = df.to_numpy().tolist()
        for idx, row in enumerate(data_rows):
            tree.insert('', 'end', values=[df.index[idx]] + row)  # Add Index and row data

        # Create vertical scrollbar
        vsb = ttk.Scrollbar(parent_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")

        # Create horizontal scrollbar
        hsb = ttk.Scrollbar(parent_frame, orient="horizontal", command=tree.xview)
        tree.configure(xscrollcommand=hsb.set)
        hsb.pack(side="bottom", fill="x")  # Horizontal scrollbar at the bottom

        tree.pack(pady=20, fill='both', expand=True)

        # Bind row click event to select a row
        tree.bind("<ButtonRelease-1>", lambda event, tree=tree, df=df: on_row_selected(event, tree, df))
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Function to handle row selection
def on_row_selected(event, tree, df):
    global details_window, selected_index
    selected_item = tree.selection()  # Get selected row
    if selected_item:
        item_values = tree.item(selected_item[0])["values"]
        selected_index = item_values[0]  # Save the Index of the selected row
        row_data = df.loc[df.index == selected_index].iloc[0]  # Extract data for the selected row
        # Close any existing details window
        if details_window:
            details_window.destroy()

        # Display a new details window
        show_row_details(row_data, df.columns)

# Function to display details of the selected row
def show_row_details(row_data, columns):
    global details_window, selected_index
    details_window = ctk.CTkToplevel(root)  # Create a new top-level window for row details
    details_window.title("Row Details")  # Window title
    width, height = 500, 400
    x = (screen_width // 2) - (width // 2)  # Center window horizontally
    y = (screen_height // 2) - (height // 2)  # Center window vertically
    details_window.geometry(f"{width}x{height}+{x}+{y}")  # Set window geometry
    details_window.configure(fg_color="white")  # Set background color

    title = ctk.CTkLabel(details_window, text="Row Details", font=("Arial", 16, "bold"))  # Title label
    title.pack(pady=10)

    # Create a scrollable frame for row details
    scroll_frame = ctk.CTkScrollableFrame(details_window, width=460, height=300, fg_color="white")
    scroll_frame.pack(pady=10, padx=20, fill="both", expand=True)

    # Display the Index of the selected row
    index_label = ctk.CTkLabel(scroll_frame, text=f"Index: {selected_index}", anchor="w", font=("Arial", 12), justify="left")
    index_label.pack(anchor="w", pady=5, padx=5)

    # Display the details of each column in the selected row
    for i, value in enumerate(row_data):
        label = ctk.CTkLabel(scroll_frame, text=f"{columns[i]}: {value}", anchor="w", font=("Arial", 12), justify="left")
        label.pack(anchor="w", pady=5, padx=5)

    # Disable "View Data" button while the details window is open
    viewDataBtn.configure(state='disabled')

    # After closing the details window, re-enable the "View Data" button
    details_window.protocol("WM_DELETE_WINDOW", on_close_details)

# Function to handle closing of details window
def on_close_details():
    global details_window
    if details_window:
        details_window.destroy()
        details_window = None
    # Re-enable "View Data" button
    viewDataBtn.configure(state='normal')

# Function to start data processing
def processing():
    global process
    process = True

# Placeholder function for visualization (currently empty)
def visualization():
    pass

# Function to display data in the given frame
def view(frame, data_to_show):
    for child in frame.winfo_children():  # Clear the current content
        child.destroy()
        root.update()
    ii = ttk.Treeview(frame)  # Create a new Treeview widget
    treeDefaults()  # Set default style for treeview
    viewTables(ii, data_to_show, frame)  # Display the data in Treeview

# Function to display data in the "View Data" page
def view_data():
    global process 
    viewDataFrame = ctk.CTkFrame(centerFrame, fg_color='transparent')
    buttonF = ctk.CTkFrame(centerFrame, fg_color='transparent')
    buttonF.pack(side='top', fill='x')

    # Display an image at the top of the "View Data" page
    img = Image.open("image2.jpg")  # Path to the image
    image_to_display = ctk.CTkImage(img, size=(800, 600))  # Resize the image
    image_label = ctk.CTkLabel(viewDataFrame,text=' ',image=image_to_display)  # Label to display the image
    image_label.pack(side="top", padx=10, pady=10)

    # Buttons for controlling data view
    dataBefore = ctk.CTkButton(buttonF, text='Data before processing', text_color='white', fg_color='black', command=lambda: view(viewDataFrame, data))
    dataAfter = ctk.CTkButton(buttonF, text='Data After processing', text_color='white', fg_color='black', command=lambda: view(viewDataFrame, dataProcessed), state='disabled' if process is None else 'normal')

    dataBefore.pack(side='left', padx=10, ipady=2)
    dataAfter.pack(side='left', padx=15, ipady=2)
    viewDataFrame.pack(side='top', expand=True, fill='both', pady=5, padx=5)

# ===================== Button Frame =====================
btnFrame = ctk.CTkFrame(root, fg_color='transparent')  # Frame to hold the buttons
btnFrame.pack(side='top', fill='x', pady=30)

# Buttons for various actions
uploadBtn = ctk.CTkButton(btnFrame, text='Upload data', text_color='white', fg_color='blue', command=lambda: switch(upload), state='normal')
uploadBtn.pack(side='left', padx=15, ipady=2)
processingBtn = ctk.CTkButton(btnFrame, text='Processing', text_color='white', fg_color='black', command=lambda: switch(processing), state='disabled')
processingBtn.pack(side='left', padx=15, ipady=2)
visualizationBtn = ctk.CTkButton(btnFrame, text='Visualization', text_color='white', fg_color='black', command=lambda: switch(visualization), state='disabled')
visualizationBtn.pack(side='left', padx=15, ipady=2)
viewDataBtn = ctk.CTkButton(btnFrame, text='View data', text_color='white', fg_color='black', command=lambda: switch(view_data), state='disabled')
viewDataBtn.pack(side='left', padx=15, ipady=2)

# ===================== Center Frame =====================
centerFrame = ctk.CTkFrame(root, fg_color='transparent')  # Frame to hold the main content
centerFrame.pack(side='top', expand=True, fill='both', pady=5, padx=5)

initialFrame()  # Show the initial frame with the image

root.mainloop()  # Start the main event loop to display the window
