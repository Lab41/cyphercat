from graphviz import Digraph

# MODEL FOOLING

def fooling_attack():
    fool_attack = Digraph('Fooling Attacks',graph_attr={'size':'8.5,11.5'},comment='Taxonomy of Secure Deep Learning', )

    fool_attack  #doctest: +ELLIPSIS

    # NODES:
    fool_attack.node('Fooling Attacks',          r'{<f0> Fooling Attacks |<f1> '+
             r'https://arxiv.org/abs/1804.00097'+
             r'\n\n}', shape='record')
    fool_attack.node('L-BFGS',          r'{<f0> L-BFGS |<f1> '+
             r'https://arxiv.org/abs/1312.6199'+
             r'\n\n}', shape='record')
    fool_attack.node('FGSM',          r'{<f0> Fast Gradient Sign \n Method (FGSM) |<f1> '+
             r'https://arxiv.org/abs/1412.6572'+
             r'\n\n}', shape='record')
    fool_attack.node('Black Box',          r'{<f0> Black Box/\n Transferable Attacks |<f1> '+
             r'https://arxiv.org/abs/1611.02770'+
             r'\n\n}' , shape = 'record')
    fool_attack.node('BIM',          r'{<f0> Basic Iterative Method(BIM)\n/ Iterative FGSM(I-FGSM) |<f1> '+
             r'https://arxiv.org/abs/1607.02533'+
             r'\n\n}', shape='record')
    fool_attack.node('PGD',          r'{<f0> Projected Gradient \n Descent |<f1> '+
             r'https://arxiv.org/abs/1706.06083'+
             r'\n\n}', shape='record')
    fool_attack.node('ATN',           r'{<f0> Adversarial Transformation \n Networks/ GANs |<f1> '+
              r'https://arxiv.org/abs/1703.09387'+
              r'\n\n}', shape='record')

    # EDGES:
    fool_attack.edge('Fooling Attacks', 'Black Box')
    fool_attack.edge('Fooling Attacks', 'White Box')
    fool_attack.edge('White Box', 'L-BFGS')
    fool_attack.edge('White Box', 'FGSM')
    fool_attack.edge('White Box', 'BIM')
    fool_attack.edge('White Box', 'PGD')
    fool_attack.edge('White Box', 'ATN')
    
    return fool_attack

#
def fooling_defense():
    fool_defense = Digraph('Fooling Defense',comment='Taxonomy of Secure Deep Learning', )
    fool_defense.edge('Fooling Defenses', 'Gradient Masking')
    fool_defense.edge('Gradient Masking', 'Complex Nonlinearities')
    fool_defense.edge('Fooling Defenses', 'Adversarial Training')
    fool_defense.edge('Fooling Defenses', 'Preprocessing')
    fool_defense.edge('Complex Nonlinearities', 'RBF Neural Networks')
    fool_defense.edge('Complex Nonlinearities', 'SVM Layers')
    fool_defense.edge('Adversarial Training', 'Data Augmentation')
    fool_defense.edge('Preprocessing', 'Noise Removal')
    return fool_defense
