"""

Module for the visualization for the confounds and their effects.

"""
import seaborn as sns


def samplets(features,
             confounds,
             x_var=None,
             y_var=None,
             group_by=None,
             color_by=None,
             sort_x_by=None,
             ):
    """
    Plot of the feature distribution for each samplet
    along a axis of a given variable,
    grouped and coloured by chosen variables.
    """

    raise NotImplementedError()


def pairs(confounds,
          color_by=None):
    """
    Plots each confound against the rest, one pair at a time, colored and grouped
    by other confounds (which are excluded in plots).

    If confounds contain Age, Gender, Disease and Site, you chose to color_by='site',
    you would have a 3x3 grid of plots:
    Age vs. Gender, vs. Disease on the first row
    Gender vs. Age, vs. Disease on the second row, and
    Disease vs. Age, vs. Gender on the third row.

    """
    pp = sns.pairplot(confounds, hue=color_by)
    return pp


def before_after():
    """
    Generic mechanism to illustrate the effect of correction or harmonization!

    Ideally in the form of a difference plot when possible.
    Or comparing before/after distributions, grouped by interesting combinations

    """

    raise NotImplementedError()
