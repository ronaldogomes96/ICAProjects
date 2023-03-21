import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Criando as variáveis de entrada e saída
angle = ctrl.Antecedent(np.arange(-45, 46, 1), 'angle')
distance = ctrl.Antecedent(np.arange(0, 11, 1), 'distance')
power = ctrl.Consequent(np.arange(0, 0.25, 0.01), 'power')

# Definindo as funções de pertinência
'''
    skfuzzy.trimf(x, [a, b, c])
    
    x: um array NumPy com os valores da variável linguística.
    [a, b, c]: uma lista ou tupla com os três parâmetros que definem a função de 
               pertinência triangular. O primeiro parâmetro a é o valor mínimo, 
               o segundo parâmetro b é o valor médio e o terceiro parâmetro c é 
               o valor máximo.
'''
angle['negative'] = fuzz.trimf(angle.universe, [-45, -45, 0])
angle['zero'] = fuzz.trimf(angle.universe, [-45, 0, 45])
angle['positive'] = fuzz.trimf(angle.universe, [0, 45, 45])

distance['near'] = fuzz.trimf(distance.universe, [0, 0, 5])
distance['medium'] = fuzz.trimf(distance.universe, [0, 5, 10])
distance['away'] = fuzz.trimf(distance.universe, [5, 10, 10])

power['low'] = fuzz.trimf(power.universe, [0, 0, 0.13])
power['medium'] = fuzz.trimf(power.universe, [0, 0.13, 0.25])
power['high'] = fuzz.trimf(power.universe, [0.13, 0.25, 0.25])

# Definindo as regras
rule_one = ctrl.Rule(distance['away'] | angle['negative'], power['high'])
rule_two = ctrl.Rule(distance['medium'], power['medium'])
rule_three = ctrl.Rule(distance['near'] | angle['positive'], power['low'])

# Criando o sistema de controle fuzzy
fuzzy_system = ctrl.ControlSystem([rule_one, rule_two, rule_three])
control = ctrl.ControlSystemSimulation(fuzzy_system)

# Definindo as entradas
control.input['angle'] = 30
control.input['distance'] = 3

# Computando o resultado
control.compute()

# Imprimindo o resultado
print(control.output) # 0.10
