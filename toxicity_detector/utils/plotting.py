import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")


def plot_category_distribution(df, categories, title, figsize=(8, 5)):
    """
    Plot a bar chart of category counts in a given DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to plot.
    categories : list
        List of categories to plot.
    title : str
        Title of the plot.
    figsize : tuple, optional
        Size of the plot figure. Defaults to (8, 5).

    Returns
    -------
    None
    """
    category_counts = [df[(df[cat] == 1)].shape[0] for cat in categories]
    
    cat_dist = pd.DataFrame({"Category": categories, "Count": category_counts})
    cat_dist = cat_dist.sort_values("Count", ascending=False)
    
    fig = cat_dist.plot(kind="bar", x="Category", y="Count", figsize=figsize, legend=False)
    plt.title(title)
    plt.xlabel("Categories")
    plt.ylabel("Count")
    fig.bar_label(fig.containers[0])
        
    plt.show()
    