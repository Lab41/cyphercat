from graphviz import Digraph

# MODEL INVERSION
def inversion_attack():
    inversion_attack = Digraph('Inversion Attacks',comment='Taxonomy of Secure Deep Learning')
    inversion_attack.node('Inversion Attacks',          r'{<f0> Inversion Attacks |<f1> '+
             r'https://arxiv.org/abs/1804.00097'+
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
    inversion_defense.edge('Inversion Defense', 'Complex Nonlinearities 1')
    inversion_defense.edge('Inversion Defense', 'Hidden Networks')
    inversion_defense.edge('Complex Nonlinearities 1', 'Gradient Lookups')
    
    return inversion_defense

