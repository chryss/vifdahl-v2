"""Shared figure style: fire class colors and slide-legible matplotlib settings."""

CLASS_COLORS = {"H": "#ffcc00", "L": "#7f3b08"}  # bright warm yellow, dark brown
CLASS_LABELS = {"H": "high (H)", "L": "low (L)"}
GROUP_LINESTYLES = ("-", "--", ":", "-.")
SLIDE_SIZE = (13.33, 7.5)  # inches, 16:9
SLIDE_RC = {"font.size": 16, "axes.titlesize": 18, "axes.labelsize": 16,
            "legend.fontsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 14,
            "axes.spines.top": False, "axes.spines.right": False}
