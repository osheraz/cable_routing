class CableTracerWrapper:
    def __init__(self, tracer, board, camera, config):
        self.tracer = tracer
        self.board = board
        self.camera = camera
        self.config = config

    def trace_cable(
        self,
        img=None,
        start_points=None,
        end_points=None,
        viz=False,
        user_pick=False,
        save=False,
    ):
        # logic from ExperimentEnv.trace_cable goes here
        pass

    def get_nearest_analytic_grasp_point(self, start_point, img=None, visualize=False):
        # logic from ExperimentEnv.get_nearest_analytic_grasp_point goes here
        pass
