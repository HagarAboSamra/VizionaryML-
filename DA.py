import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    return pd.read_csv(r"F:\data analysis\first_project\Fitness_trackers.csv")


def get_categorical_columns(df):
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def get_numeric_columns(df):
    return df.select_dtypes(include='number').columns.tolist()


def validate_column(df, column, numeric_required=False):
    if column not in df.columns:
        print("❌ Invalid column name.")
        return False
    if numeric_required and not pd.api.types.is_numeric_dtype(df[column]):
        print("❌ Column must be numeric.")
        return False
    return True


def choose_column(df, column_type, numeric_required=False):
    if column_type == 'categorical':
        cols = get_categorical_columns(df)
    elif column_type == 'numeric':
        cols = get_numeric_columns(df)

    if not cols:
        print(f"❌ No {column_type} columns found.")
        return None

    print(f"✅ Available {column_type} columns:", cols)


    for idx, col in enumerate(cols):
        print(f"{idx + 1}. {col}")

    choice = input(f"Enter the number of the {column_type} column: ")

    if choice.isdigit() and 1 <= int(choice) <= len(cols):
        return cols[int(choice) - 1]
    else:
        print("❌ Invalid choice.")
        return None


def create_pie_chart(df):
    column = choose_column(df, 'categorical')
    if column is None:
        return
    top_categories = df[column].value_counts().head(5)
    plt.pie(
        top_categories.values,
        labels=top_categories.index.tolist(),
        autopct='%1.1f%%',
        radius=1,
        explode=[0.05] * len(top_categories),
        shadow=True,
        textprops={'fontsize': 14}
    )
    plt.title(f"Top 5 in '{column}'", fontsize=16)
    plt.show()


def create_bar_chart(df):
    column = choose_column(df, 'categorical')
    if column is None:
        return
    df[column].value_counts().plot(kind="bar", color='skyblue', edgecolor='black')
    plt.title(f"{column} Frequency")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def create_horizontal_bar_chart(df):
    column = choose_column(df, 'categorical')
    if column is None:
        return
    value_counts = df[column].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(value_counts.index, value_counts.values, color='steelblue', edgecolor='black')
    plt.xlabel("Count")
    plt.ylabel(column)
    plt.title(f"Top 10 {column}")
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()


def create_stripplot(df):
    x_col = choose_column(df, 'categorical')
    y_col = choose_column(df, 'numeric', numeric_required=True)
    if x_col is None or y_col is None:
        return
    plt.figure(figsize=(16, 10))
    sns.stripplot(x=x_col, y=y_col, data=df, jitter=True)
    plt.title(f"{y_col} vs {x_col}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def create_countplot(df):
    primary = choose_column(df, 'categorical')
    hue = choose_column(df, 'categorical')
    if primary is None or hue is None:
        return
    plt.figure(figsize=(10, 7))
    sns.countplot(y=primary, hue=hue, data=df,
                  order=df[primary].value_counts().iloc[:10].index)
    plt.title(f"Top 10 {primary} by {hue}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def create_pairplot(df):
    num_cols = get_numeric_columns(df)
    cat_cols = get_categorical_columns(df)
    if not num_cols:
        print("❌ No numeric columns found.")
        return
    print("✅ Numeric columns:", num_cols)
    hue = input("Hue column (press Enter to skip): ")
    if hue and hue not in df.columns:
        print("❌ Invalid hue column.")
        return
    sns.set_style("ticks")
    cols = num_cols + [hue] if hue else num_cols
    sns.pairplot(df[cols], hue=hue if hue else None)
    plt.show()


def hist_plot(df):
    column = choose_column(df, 'numeric', numeric_required=True)
    if column is None:
        return
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], kde=True, color='cornflowerblue', edgecolor='black')
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def box_plot(df):
    column = choose_column(df, 'numeric', numeric_required=True)
    if column is None:
        return
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[column], color='skyblue')
    plt.title(f"Boxplot of {column}")
    plt.xlabel(column)
    plt.grid(True)
    plt.show()


def create_heatmap(df):
    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        plt.figure(figsize=(10, 7))
        sns.heatmap(numeric_df.corr(), annot=True, square=True, cmap='coolwarm', linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()
    else:
        print("No numeric data available to plot heatmap.")


def create_kde_plot(df):
    column = choose_column(df, 'numeric', numeric_required=True)
    if column is None:
        return
    plt.figure(figsize=(8, 5))
    sns.kdeplot(df[column].dropna(), fill=True, color='purple', linewidth=2)
    plt.title(f"KDE Plot of {column}")
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def create_scatter_plot(df):
    x_col = choose_column(df, 'numeric', numeric_required=True)
    y_col = choose_column(df, 'numeric', numeric_required=True)
    if x_col is None or y_col is None:
        return
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, color='teal')
    plt.title(f"{y_col} vs {x_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    df = load_data()

    while True:
        print("\nSelect a visualization to generate:")
        print("1. Pie Chart")
        print("2. Bar Chart")
        print("3. Horizontal Bar Chart")
        print("4. Strip Plot")
        print("5. Count Plot")
        print("6. Pair Plot")
        print("7. Histogram Plot")
        print("8. Box Plot")
        print("9. Heatmap")
        print("10. KDE Plot")
        print("11. Scatter Plot")
        print("0. Exit")

        choice = input("Enter your choice (0-11): ")

        if choice == '1':
            create_pie_chart(df)
        elif choice == '2':
            create_bar_chart(df)
        elif choice == '3':
            create_horizontal_bar_chart(df)
        elif choice == '4':
            create_stripplot(df)
        elif choice == '5':
            create_countplot(df)
        elif choice == '6':
            create_pairplot(df)
        elif choice == '7':
            hist_plot(df)
        elif choice == '8':
            box_plot(df)
        elif choice == '9':
            create_heatmap(df)
        elif choice == '10':
            create_kde_plot(df)
        elif choice == '11':
            create_scatter_plot(df)
        elif choice == '0':
            print("Exiting. Goodbye 👋")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
