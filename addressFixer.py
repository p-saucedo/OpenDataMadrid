class AddresFixer:

    def __init__(self):
        pass

    
    def fixAddress(s):
        traducciones = {
        "CALL." : "CALLE",
        "C/" : "CALLE",
        "AVDA." : "AVENIDA",
        "AV " : "AVENIDA ",
        "POLIG." : "POLIGONO",
        "PASEO." : "PASEO",
        "CTRA." : "CARRETERA",
        "PLAZA." : "PLAZA",
        "GTA." : "GLORIETA",
        "PQUE." : "PARQUE",
        "AUTOV." : "AUTOVIA",
        "CUSTA." : "CUESTA",
        "CMNO." : "CAMINO",
        "C�ADA.": "CAÑADA"
        }
        #print("Original: {}".format(s))
        for f_key, f_value in traducciones.items():
            s = s.replace(f_key, f_value)
        index = s.find('/')
        if index == -1:
        #    print("Final: {}".format(s))
            return s
        else:
            ss = s[index-1:]
            s = s.replace(ss,"")
        #    print("Final: {}".format(s))
            return s
        

    