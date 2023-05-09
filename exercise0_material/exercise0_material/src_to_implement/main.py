import matplotlib.pyplot as plt
from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator

checker_board = Checker(500, 50)
checker_board.draw()
checker_board.show()

circle = Circle(1024, 200, (512, 256))
circle.draw()
circle.show()

spectrum = Spectrum(300)
spectrum.draw()
spectrum.show()


data_path = './data/exercise_data'
label_path = './data/Labels.json'
image_gen = ImageGenerator(data_path, label_path, 40, [32, 32, 3], rotation=False, mirroring=False,shuffle=False)
image_gen.show()