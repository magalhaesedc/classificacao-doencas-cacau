def printTime(title, begin, end):

    duracao = calcTime(end-begin)
    num_caracteres = 50
    metade_caracteres = (((num_caracteres - len(title)) // 2) - 2)
    num_caracteres = ((metade_caracteres * 2) + 2 + len(title))
    print(f"\n{metade_caracteres*'='} {title} {metade_caracteres*'='}")
    print("\nTempo de Execução:", duracao)
    print(f"\n{num_caracteres*'='}\n")


def calcTime(segundos):
    h = 0
    m  = 0
    
    if(segundos >= 3600):
        h = segundos // 3600
        segundos = segundos - (h * 3600)
    
    if(segundos >= 60):
        m = segundos // 60
        segundos = segundos - (m * 60)
    
    if(h > 0):
        return str(int(h))+"h "+str(int(m))+"min "+str(round(segundos, 2))+"s"
    elif(m > 0):
        return str(int(m))+"min "+str(round(segundos, 2))+"s"
    else:
        return str(round(segundos, 2))+"s"