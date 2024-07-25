from perception import object_detector


class Planner:
    def __init__(self, simulator, object_detector: object_detector.ObjectDetector, nominal_controller) -> None:
        self.simulator = simulator
        self.object_detector = object_detector
        self.nominal_controller = nominal_controller

    def add_new_text_query(self, query):
        self.object_detector.add_new_text_query(query)
