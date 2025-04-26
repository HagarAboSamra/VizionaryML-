import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

"""
Message for GUI developer

This file contains the show_plot_in_frame() function in line 200
with parameters:
1. frame : The frame in which the drawing will be shown
2. df : The DataFrame
3. col_name : The column(s) to be drawn (can be a string or a list/tuple)
4. mode : Type of plot
"""

# ========== Helper Functions ==========


# Return list of categorical columns in the DataFrame
def get_categorical_columns(df):
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


# Return list of numeric columns in the DataFrame
def get_numeric_columns(df):
    return df.select_dtypes(include="number").columns.tolist()


# Validate if a column exists and matches expected type
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


# ========== Plot Functions ==========


# Pie chart of top 5 most frequent values in a categorical column
def create_pie_chart(ax, df, column_name):
    column = choose_column(df, column_name, "categorical")
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


# Bar chart of value frequencies for a categorical column
def create_bar_chart(ax, df, column_name):
    column = choose_column(df, column_name, "categorical")
    if column is None:
        return
    value_counts = df[column].value_counts()
    ax.bar(value_counts.index, value_counts.values, color="skyblue", edgecolor="black")
    ax.set_title(f"{column} Frequency")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.tick_params(axis="x", rotation=45)


# Horizontal bar chart of top 10 values in a categorical column
def create_horizontal_bar_chart(ax, df, column_name):
    column = choose_column(df, column_name, "categorical")
    if column is None:
        return
    value_counts = df[column].value_counts().head(10)
    ax.barh(
        value_counts.index, value_counts.values, color="steelblue", edgecolor="black"
    )
    ax.set_title(f"Top 10 {column}")
    ax.set_xlabel("Count")
    ax.set_ylabel(column)


# Strip plot between one categorical and one numeric column
def create_stripplot(ax, df, categorical_col, numeric_col):
    if categorical_col is None or numeric_col is None:
        return
    sns.stripplot(x=categorical_col, y=numeric_col, data=df, jitter=True, ax=ax)
    ax.set_title(f"{numeric_col} vs {categorical_col}")


# Count plot (grouped bar chart) for two categorical columns
def create_countplot(ax, df, primary_col, hue_col):
    if primary_col is None or hue_col is None:
        return
    sns.countplot(
        y=primary_col,
        hue=hue_col,
        data=df,
        order=df[primary_col].value_counts().iloc[:10].index,
        ax=ax,
    )
    ax.set_title(f"Top 10 {primary_col} by {hue_col}")


# Pair plot for all numeric columns, using first categorical column as hue if available
def create_pairplot(_, df):
    num_cols = get_numeric_columns(df)
    hue = get_categorical_columns(df)[0] if get_categorical_columns(df) else None
    sns.pairplot(df[num_cols + [hue]] if hue else df[num_cols], hue=hue)
    plt.show()


# Histogram for a numeric column with optional KDE
def hist_plot(ax, df, column_name):
    column = choose_column(df, column_name, "numeric", numeric_required=True)
    if column is None:
        return
    sns.histplot(df[column], kde=True, color="cornflowerblue", edgecolor="black", ax=ax)
    ax.set_title(f"Histogram of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")


# Box plot for a single numeric column
def box_plot(ax, df, column_name):
    column = choose_column(df, column_name, "numeric", numeric_required=True)
    if column is None:
        return
    sns.boxplot(x=df[column], color="skyblue", ax=ax)
    ax.set_title(f"Boxplot of {column}")
    ax.set_xlabel(column)


# Heatmap of correlation matrix between numeric columns
def create_heatmap(ax, df):
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


# KDE (Kernel Density Estimation) plot for numeric column
def create_kde_plot(ax, df, column_name):
    column = choose_column(df, column_name, "numeric", numeric_required=True)
    if column is None:
        return
    sns.kdeplot(df[column].dropna(), fill=True, color="purple", linewidth=2, ax=ax)
    ax.set_title(f"KDE Plot of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Density")


# Scatter plot between two numeric columns
def create_scatter_plot(ax, df, x_column, y_column):
    if x_column is None or y_column is None:
        return
    sns.scatterplot(data=df, x=x_column, y=y_column, color="teal", ax=ax)
    ax.set_title(f"{y_column} vs {x_column}")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)


# ========== Display Plot in GUI Frame ==========


# Main function to display any selected plot inside a CustomTkinter frame
def show_plot_in_frame(frame, df, col_name, mode):
    # Mapping plot modes to their drawing functions
    draw_functions = {
        "Pie Chart": create_pie_chart,
        "Bar Chart": create_bar_chart,
        "Horizontal Bar Chart": create_horizontal_bar_chart,
        "Strip Plot": create_stripplot,
        "Count Plot": create_countplot,
        "Pair Plot": create_pairplot,
        "Histogram Plot": hist_plot,
        "Box Plot": box_plot,
        "Heatmap": create_heatmap,
        "KDE Plot": create_kde_plot,
        "Scatter Plot": create_scatter_plot,
    }

    # Clear any previous plots in the frame
    for widget in frame.winfo_children():
        widget.destroy()

    # Handle special case: pairplot (opens in a separate window)
    if mode == "Pair Plot":
        draw_functions[mode](None, df)
        return

    # Calculate frame dimensions and setup matplotlib figure
    frame.update_idletasks()
    width_px = frame.winfo_width()
    height_px = frame.winfo_height()
    dpi = 100
    fig = Figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_subplot(111)

    # Handle single or multiple columns depending on plot type
    if isinstance(col_name, (list, tuple)):
        draw_functions[mode](ax, df, *col_name)
    else:
        draw_functions[mode](ax, df, col_name)

    # Display the matplotlib figure inside the Tkinter frame
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
