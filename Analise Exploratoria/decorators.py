import matplotlib.pyplot as plt
import seaborn as sns


def rotulos(func):
    def wrapper(*args, **kwargs):

        grafico = func(*args, **kwargs)
        
        for container in grafico.containers:
            grafico.bar_label(container, label_type="edge", color="black",
                        padding=6,
                        fontsize=9,
                        bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white', 'edgecolor': 'black'})
        plt.xlabel('')
        plt.ylabel('')
        plt.show()
    return wrapper



def resize(alt, larg):
    def decorator(func):
        def wrapper(*args, **kwargs):
            
            plt.subplots(figsize=(alt,larg))
            func(*args, **kwargs)

        return wrapper
    return decorator



def show():
    def decorator(func):
        def wrapper(*args, **kwargs):
            
            func(*args, **kwargs)
            plt.xlabel('')
            plt.ylabel('')
            plt.show()

        return wrapper
    return decorator