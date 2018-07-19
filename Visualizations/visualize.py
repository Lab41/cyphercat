from graphviz import Digraph
from model_inversion import *
from model_fooling import *
from visualize_util import *

# Generate the PDF's
visualize(fooling_attack(), 'fooling_attack.gv')
visualize(fooling_defense(), 'fooling_defense.gv')
visualize(inversion_attack(), 'inversion_attack.gv')
visualize(inversion_attack(), 'inversion_defense.gv')