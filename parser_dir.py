class ParserDir:

    def __init__(self):
        pass

    
    def parsearDireccion(self, direccion):

        # Lo suyo seria utilizar expresiones regulares
        pos = direccion.find("CALL.")
        posfin= direccion.rfind("CALL.")
        #print(direccion)
        #if ( pos != -1):
        #    pass
        if( posfin != -1 ):
            pass
        direccion = direccion.replace("CALL.", "CALLE")
        #print(direccion)