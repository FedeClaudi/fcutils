def forceAspect(ax, aspect):
    """
        Forces the aspect ratio of an axis onto which an image what plotted with imshow.
        From:
        https://www.science-emergence.com/Articles/How-to-change-imshow-aspect-ratio-in-matplotlib-/
    """
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(
        abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect
    )


def make_axes_at_center(ax):
    """
        Moves the axes splines to the center instead of the bottom left of the ax
        from https://stackoverflow.com/questions/31556446/how-to-draw-axis-in-the-middle-of-the-figure
    """
    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines["left"].set_position("center")
    ax.spines["bottom"].set_position("center")

    # Eliminate upper and right axes
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
