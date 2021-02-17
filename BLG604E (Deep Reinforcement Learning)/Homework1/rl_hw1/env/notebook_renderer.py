""" Ipython/Jupyter notebook renderer for gymcolab environments.
    Author: Tolga Ok & Nazim Kemal Ure
"""

from collections import defaultdict
from ipycanvas import Canvas, hold_canvas
import matplotlib


class CanvasRenderer():
    """ Ipycanvas renderer for jupyter notebook. Note that, currently doesn't
    work with the jupyter-lab. Renderer draws the gymcolab environment board
    at each call. For each cropper a board is drawn side to side. This renderer
    requires an initialization before the environment loop. So, after the
    environment is initialized ```env.init_render()``` must be called.
    Arguments:
        - cell_size: Size of each cell in terms of pixel
        - colors: Color dictionary to map each character to its corresponding
            color. If a character is not defined in the dictionary the default
            color is used.
        - croppers: List of cropper objects to be rendered.
        - border_ration: Ratio of the empty space at the border to the cell
            size
    """

    DEFAULT_COLOR = "#CCCCCC"

    def __init__(self, cell_size, colors, croppers, border_ratio=0.05):
        self.croppers = sorted(croppers, key=lambda x: x.rows, reverse=True)
        width = ((sum(cropper.cols for cropper in croppers) +
                  len(croppers) - 1) * cell_size)
        height = max(cropper.rows for cropper in croppers) * cell_size

        self.canvas = Canvas(height=height*cell_size, width=width*cell_size)
        self.border_ratio = border_ratio
        self.colors = defaultdict(lambda: self.DEFAULT_COLOR)
        for key, value in colors.items():
            self.colors[ord(key)] = value
        self.cell_size = cell_size

    def __call__(self, board, y_offset=0, x_offset=0, cmap=None):
        """ Render the board using croppers.
            Raise:
                - Attrubute Error: If the renderer is not initialized using
                    <_init_render> function
        """
        
        cropper = self.croppers[0]
        for ix, cropper in enumerate(self.croppers):
            board = cropper.crop(board).board
            height, width = board.shape
            self.draw(board, 0, self.cell_size * (width + 1) * ix, None)

    def draw(self, board, y_offset, x_offset, cmap):
        border = self.cell_size * self.border_ratio
        canvas = self.canvas
        with hold_canvas(canvas):
            height, width = board.shape
            for iy in range(height):
                for ix in range(width):
                    if cmap is None:
                        canvas.fill_style = self.colors[board[iy, ix]]
                    else:
                        color = cmap(board[iy, ix])[:3]
                        canvas.fill_style = matplotlib.colors.rgb2hex(color)
                    canvas.fill_rect(x_offset + ix * self.cell_size + border,
                                     y_offset + iy * self.cell_size + border,
                                     self.cell_size - border * 2,
                                     self.cell_size - border * 2)
