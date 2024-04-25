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



def elim_gaussiana():
    A = np.array([[2,1,2,3],[4,3,3,4],[-2,2,-4,-12],[4,1,8,-3]])
    Ac = A.copy()
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    for d in range(1,m):
        pivot = A[d-1][d-1]
        for i in range(1,m):
            k = Ac[i][d-1]/pivot
            A[i][d-1] = k
            for j in range (1, n):
                A[i][j] = Ac[i][j] - k*Ac[i-1][j]
                cant_op += 1
    return (A, cant_op) 
print(elim_gaussiana())
