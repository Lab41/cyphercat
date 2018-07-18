from graphviz import Digraph

dot1 = Digraph('structs',comment='Taxonomy of Secure Deep Learning')

dot1  #doctest: +ELLIPSIS

# MODEL FOOLING
#
dot1.node('Fooling Attacks', \
         r'{<f0> Fooling Attacks |<f1> '+
         r'https://arxiv.org/abs/1804.00097'+
         r'\n\n}', shape='record')
dot1.edge('Fooling Attacks', 'Transferable Attacks')

# 
dot1.node('Fooling Defenses', \
         r'{<f0> Fooling Defenses |<f1> '+
         r'https://arxiv.org/abs/1804.00097'+
         r'\n\n}', shape='record')
dot1.edge('Fooling Defenses', 'Complex Nonlinearities')
dot1.edge('Fooling Defenses', 'Adversarial Training')
dot1.edge('Complex Nonlinearities', 'RBF Neural Networks')
dot1.edge('Complex Nonlinearities', 'SVM Layers')
dot1.edge('Adversarial Training', 'Data Augmentation')


# MODEL INVERSION
#
dot2 = Digraph('structs',comment='Taxonomy of Secure Deep Learning')
dot2.node('Inversion Attacks', \
         r'{<f0> Inversion Attacks |<f1> '+
         r'https://arxiv.org/abs/1804.00097'+
         r'\n\n}', shape='record')


dot2.edge('Inversion Attacks', 'Membership Inference')
dot2.edge('Membership Inference', 'Shadow Networks')
dot2.edge('Inversion Attacks', 'Data Extraction')
dot2.edge('Data Extraction', 'GAN Networks')
dot2.edge('Data Extraction', 'Gradient Ascent')
dot2.edge('Inversion Attacks', 'Model Extraction')

#
dot2.edge('Inversion Defense', 'Complex Nonlinearities 1')
dot2.edge('Inversion Defense', 'Hidden Networks')
dot2.edge('Complex Nonlinearities 1', 'Gradient Lookups')

print(dot1.source)  # doctest: +NORMALIZE_WHITESPACE
dot1.render('./fooling-table.gv', view=True)  # doctest: +SKIP
'./fooling.gv.pdf'

print(dot2.source)  # doctest: +NORMALIZE_WHITESPACE
dot2.render('./inversion-table.gv', view=True)  # doctest: +SKIP
'./inversion.gv.pdf'

