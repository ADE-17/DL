class BaseLayer:
    """
    Simple base layer with argument if the layer is trainable or not
    """
    def __init__(self):
        self.trainable = False #Set False
        self.testing_phase = False #Set False
    