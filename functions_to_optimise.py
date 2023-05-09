import numpy as np 
import math as mt

def COBB_DOUGLAS(input):
    """
    This function is a toy to test the pso with a simple function to create 
    
    The input is a numpy with all the inputs of the function : if the function 
    go from R^3 to R, we will have a numpy on 3 dim to another output array with a unique value

    Here, we will take the function : COBB douglas f(input) = 3*((input[0])**(1/4))*((input[1])**(3/4))
    
    -     minx = 0
    -     maxx = 15 
    
    Args:
        input (np.array): contains all the inputs of the function here we have 2 variables (shape of input 2x1)
        
    Returns:
        np.array : contains all the outputs of the function 
    """
    return np.array(3*((input[0])**(1/4))*((input[1])**(3/4))) 


def Schwefel(input):
    """
    Here, we will take the function : Schwefel f(input) = 418.9829*2 - input[0]*np.sin(abs(input[0])**0.5) - input[1]*np.sin(abs(input[1])**0.5)
    
    
    -    minx = -500
    -    maxx = 500
    
    Args:
        input (np.array): contains all the inputs of the function here we have 2 variables (shape of input 2x1)
        
    Returns:
        np.array : contains all the outputs of the function 
    """
    return np.array(418.9829*2 - input[0]*np.sin(abs(input[0])**0.5) - input[1]*np.sin(abs(input[1])**0.5)) 


def Ackley(input) : 
    """
    Here, we will take the function : Ackley(input) = -Aexp(-Bsqrt((1/n)*sum(xi^2))) - exp((1/n)sum(cos(Cxi))) + A + exp(1)
    
     xi ∈ [-32.768, 32.768]

    Args:
        input (np.array): contains all the inputs of the function here we have 2 variables (shape of input 2x1)
        
    Returns:
        np.array : contains all the outputs of the function 
    """
    a = 1 
    b = 100 
    return np.array((a - input[0])**2 + b*(input[1] - input[0]**2)**2) 

def Rastrigin(input):
    """
    Here, we will take the function : Rastrigin(input) = 10*2 + (input[0]**2-10*np.cos(2*mt.pi*input[0])) + (input[1]**2-10*np.cos(2*mt.pi*input[1]))

    -    minx = -5
    -    maxx = 5
    
    Args:
        input (np.array): contains all the inputs of the function here we have 2 variables (shape of input 2x1)
        
    Returns:
        np.array : contains all the outputs of the function 
    """
    return np.array(10*2 + (input[0]**2-10*np.cos(2*mt.pi*input[0])) + (input[1]**2-10*np.cos(2*mt.pi*input[1]))) 



def Rosenbrock(input):
    """
    Here, we will take the function : Rosenbrock(input) = 100*((input[1]-input[0]**2)**2) + (input[0]- 1)**2 

    -     minx = -10
    
    -     maxx = 10
    
    Args:
        input (np.array): contains all the inputs of the function here we have 2 variables (shape of input 2x1)
        
    Returns:
        np.array : contains all the outputs of the function 
    """
    return np.array(100*((input[1]-input[0]**2)**2) + (input[0]- 1)**2) 




def Easom(input):
    """
    Here, we will take the function : Easom(input) = -np.cos(input[0])*np.cos(input[1])*np.exp(-(input[0]-mt.pi)**2-(input[1]-mt.pi)**2)

    Args:
        input (np.array): contains all the inputs of the function here we have 2 variables (shape of input 2x1)
        
    Returns:
        np.array : contains all the outputs of the function 
    """
    return np.array(-np.cos(input[0])*np.cos(input[1])*np.exp(-(input[0]-mt.pi)**2-(input[1]-mt.pi)**2)) 


def Function_test_1(input):
    """
    Here, we will take the function : Additive(input) = -.4 + (x+15)/30. + (y+15)/40.+.5*np.sin(r)
    
    with x,y in [-15,15]

    Args:
        input (np.array): contains all the inputs of the function here we have 2 variables (shape of input 2x1)
        
    Returns:
        np.array : contains all the outputs of the function 
    """
    x = input[0]
    y = input[1]
    r = np.sqrt(x**2 + y**2)
    return -.4 + (x+15)/30. + (y+15)/40.+.5*np.sin(r)

def Additive(input):
    """
    Here, we will take the function : Additive(input) =  input[1]**2 + input[0]**2 + 2

    Args:
        input (np.array): contains all the inputs of the function here we have 2 variables (shape of input 2x1)
        
    Returns:
        np.array : contains all the outputs of the function 
    """
    return np.array(input[1]**2 + input[0]**2 + 2) 

