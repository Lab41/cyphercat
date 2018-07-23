from graphviz import Digraph

# MODEL INVERSION
def inversion_attack():
    inversion_attack = Digraph('Inversion Attacks',comment='Taxonomy of Secure Deep Learning')
    inversion_attack.node('Inversion Attacks',          r'{<f0> Inversion Attacks |<f1> '+
             r'https://arxiv.org/abs/1804.00097'+
             r'\n\n}', shape='record')
    inversion_attack.node('Shadow Networks',            r'{<f0> Shadow Networks |<f1> '+
             r'https://arxiv.org/abs/1610.05820\n'+
             r'https://arxiv.org/abs/1806.01246'+
             r'\n\n}', shape='record')
    inversion_attack.node('Data Extraction',            r'{<f0> Data Extraction |<f1> '+
             r'https://arxiv.org/abs/1806.00400v1'+
             r'\n\n}', shape='record')

    inversion_attack.edge('Inversion Attacks', 'Membership Inference')
    inversion_attack.edge('Membership Inference', 'Shadow Networks')
    inversion_attack.edge('Inversion Attacks', 'Data Extraction')
    inversion_attack.edge('Data Extraction', 'GAN Networks')
    inversion_attack.edge('Data Extraction', 'Gradient Ascent')
    inversion_attack.edge('Inversion Attacks', 'Model Extraction')
    
    return inversion_attack

def inversion_defense():
    inversion_defense = Digraph('Inversion Defenses',comment='Taxonomy of Secure Deep Learning')
    inversion_defense.node('Adversarial Training', r'{<f0> Adversarial Training |<f1> '+
             r'https://arxiv.org/abs/1807.05852'+
             r'\n\n}', shape='record')

    inversion_defense.edge('Inversion Defense', 'Complex Nonlinearities')
    inversion_defense.edge('Inversion Defense', 'Hidden Networks')
    inversion_defense.edge('Inversion Defense', 'Adversarial Training')
    inversion_defense.edge('Complex Nonlinearities', 'Gradient Lookups')
    
    return inversion_defense

