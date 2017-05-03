from process_data import *
import matplotlib.pyplot as plt
import seaborn as sns


def visual(train):

    # tabulate
    train[['Sex', 'Class']].groupby(['Sex'],as_index=False).mean().sort_values(by='Class',ascending=False)
    train[['Job', 'Class']].groupby(['Job'],as_index=False).mean().sort_values(by='Class',ascending=False)
    train[['Housing', 'Class']].groupby(['Housing'],as_index=False).mean().sort_values(by='Class',ascending=False)
    train[['Saving accounts', 'Class']].groupby(['Saving accounts'],as_index=False).mean().sort_values(by='Class',ascending=False)
    train[['Checking account', 'Class']].groupby(['Checking account'],as_index=False).mean().sort_values(by='Class',ascending=False)
    train[['Purpose', 'Class']].groupby(['Purpose'],as_index=False).mean().sort_values(by='Class',ascending=False)
    train[['Purpose', 'Class']].groupby(['Purpose'],as_index=False).mean().sort_values(by='Class',ascending=False)

    # visualize
    g = sns.FacetGrid(train, col='Class')
    g.map(plt.hist, 'Age', bins=20)
    # similar for Good and Bad

    grid = sns.FacetGrid(train, col='Class', row='Housing', size=2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend()

    grid = sns.FacetGrid(train, col='Class', row='Sex', size=2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend()

    grid = sns.FacetGrid(train, row='Purpose', size=2.2, aspect=1.6)
    grid.map(sns.pointplot, 'Housing', 'Class', 'Sex', palette='deep')
    grid.add_legend()

    grid = sns.FacetGrid(train, col='Saving accounts', row='Checking account', size=2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend()

    grid = sns.FacetGrid(train, row='Saving accounts', size=2.2, aspect=1.6)
    grid.map(plt.hist, 'Credit amount', bins=20)
    grid.add_legend()

    sns.factorplot('Checking account','Class', data=train,size=4,aspect=3)
    # 1 - 2: missing, rich, moderate, little

    sns.countplot(x='Saving accounts', hue='Class', data=train, size=4, aspect=3)
    # 1 - 2: quite rich, rich, moderate, little


    grid = sns.FacetGrid(train, col='Class', row='Checking account', size=2.2, aspect=1.6)
    grid.map(plt.hist, 'Credit amount', alpha=.5, bins=20)
    grid.add_legend()

    # only when checking account info is missing, low credit amount - more in class 1

    g = sns.FacetGrid(train, col='Saving accounts')
    g.map(plt.hist, 'Age', bins=20)
    # shift to older

    g = sns.FacetGrid(train, col='Saving accounts', row='Class')
    g.map(plt.hist, 'Age', bins=20)
    #

    sns.factorplot('Credit amount', 'Class', data=train, size=4, aspect=3)

    sns.regplot(x="Credit amount", y="Class", data=train);
    # 1 - 2: quite rich, rich, moderate, little

    sns.regplot('Age', 'Class', data=train, size=4, aspect=3)

    sns.regplot(x="Age", y="Class", data=train);
    # 1 - 2: quite rich, rich, moderate, little
    sns.plt.show()

    from collections import Counter
    Counter(train['Saving accounts'])

    with sns.plotting_context("notebook",font_scale=1.5):
        sns.set_style("whitegrid")
        sns.distplot(train["Age"], bins=80,
                     kde=False,
                     color="red")
        sns.plt.title("Age Distribution")
        plt.ylabel("Count")


