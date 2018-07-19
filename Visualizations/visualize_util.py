from graphviz import Digraph

def visualize(model, name):
    print(model.source)  # doctest: +NORMALIZE_WHITESPACE
    model.render(name, view=False)  # doctest: +SKIP
    #name.pdf
    
# print(fool_attack.source)  # doctest: +NORMALIZE_WHITESPACE
# fool_attack.render('test-output/fooling_attack-table.gv', view=False)  # doctest: +SKIP
# #'test-output/fooling_attack.gv.pdf'

# print(fool_defense.source)  # doctest: +NORMALIZE_WHITESPACE
# fool_defense.render('test-output/fooling_defense-table.gv', view=False)  # doctest: +SKIP
# #'test-output/fooling_defense.gv.pdf'

# print(inversion_attack.source)  # doctest: +NORMALIZE_WHITESPACE
# inversion_attack.render('test-output/inversion_attack-table.gv', view=False)  # doctest: +SKIP
# #'test-output/inversion_attack.gv.pdf'

# print(inversion_defense.source)  # doctest: +NORMALIZE_WHITESPACE
# inversion_defense.render('test-output/inversion_defense-table.gv', view=False)  # doctest: +SKIP
# #'test-output/inversion_defense.gv.pdf'