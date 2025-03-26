class Board:

    def __init__(
        self,
        config_path="/home/osheraz/cable_routing/cable_routing/configs/board/board_config.json",
        width=1500,
        height=800,
        grid_size=(100, 100),
    ):

        self.config_path = config_path
        self.true_width = width  # Stores the width in pixels
        self.true_height = height  # Stores the height in pixels

        self.width = -(width // -grid_size[0])
        self.height = -(height // -grid_size[1])
        self.grid_size = grid_size

        self.cables = {}
        self.board = [["." for _ in range(self.width)] for _ in range(self.height)]

        self.clip_positions = self.load_board_config()
        self.point1, self.point2 = (582, 5), (1391, 767)  # ROI

    def load_board_config():
        
