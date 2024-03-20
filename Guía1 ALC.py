#Ejercicio 21
import numpy as np
def traza(A)-> int:
    pos = 0
    traza = 0
    for elem in A:
        traza += elem[pos]
        pos += pos + 1
    return traza 

def sum_matriz(A)-> int:
    suma = 0
    for elem in A:
        for i in range(len(elem)):
            suma += elem[i]
    return suma

def suma_pos_o_neg(A)-> bool:
    sumaP = 0
    sumaN = 0
    for elem in A:
        for i in range(len(elem)):
            if elem[i] > 0:
                sumaP += elem[i]
            else:
                sumaN += (-1) * elem[i]
    if sumaP > sumaN:
        return True
    else:
        return False