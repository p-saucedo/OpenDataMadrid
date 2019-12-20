# OpenDataMadrid #

El objetivo de esta aplicación es utilizar los conjuntos de datos públicos que el Ayuntamiento de Madrid mantiene actualizados [aquí](https://datos.madrid.es/portal/site/egob/). La aplicación ofrece el tratamiento de esos datos, dándoles un formato adecuado para que despueś el motor de machine learning desarrollado pueda tratarlos y ,con ello, predecir resultados. La aplicación es muy flexible y permite el uso de diversos datasets con distintos objetivos.

### Nuestro objeto de estudio ###

En esta primera versión, haremos uso de los datos sobre los accidentes con implicación de bicicletas en 2019 para predecir dos cosas: por un lado, la probabilidad de accidente en cualquier punto de la ciudad y, por el otro, la gravedad del accidente en caso de que este se produjese. En el dataset utilizado, cada fila corresponde a un accidente en el que se especifica mucha información. Nos centraremos en dónde se ha producido el accidente y con qué gravedad.

#### Análisis de los datos ####

La primera parte consiste en coger los conjuntos de datos y tratarlos. Cada uno es un mundo y tiene un formato distinto, por lo que la aplicación debe de ser flexible y poder adaptarse dependiendo de dónde esté la información necesaria. Para este objetivo, el usuario podrá definir el separador del csv que va a subir, la columna donde se encuentra la dirección y si la dirección está en formato dirección o coordenadas. Estos parámetros serán necesarios para que la aplicación trabaje con los datos.

#### Traducción de direcciones ####

Si la dirección del accidente se especifica en coordenadas, el proceso resulta algo más sencillo. Por el contrario, si la dirección viene dado como un texto, este viene expresado con abreviaturas (*CALL*, *AVDA*,...) y además, en ciertas ocasiones, la direccioń que se especifica es una esquina (*CALL X / CALL Y*). Las abreviaturas se traducen y las esquinas se transforman en la dirección de la primera calle, para que la API de geolocalización las entienda.

Una vez que se tiene la dirección en formato texto, se deben geolocalizar las direcciones para obtener sus coordenadas. Para ello haremos uso de la libreria [Geopy](https://geopy.readthedocs.io/en/stable/) que nos ofrece una API que permite utilizar distintos geolocalizadores de una manera sencilla. En concreto, OpenDataMadrid usa [Bing Maps](https://www.bing.com/maps) que es un servicio que permite multitud de peticiones, con un tiempo de respuesta bajo y con una precisión bastante elevada. Con esta API podremos geolocalizar todas las direcciones y obtener su latitud y su longitud.


#### Mapa y sus posiciones ####

La aplicación está montada en Flask y hace uso de la libreria de javascript [Leaflet](https://leafletjs.com/) para mostrar el mapa y permitir interactuar con él (hacer zoom, desplazarse, hacer click...). Este mapa recibe la posiciones a mostrar en formato GeoJSON. Este formato tiene una sintaxis muy similar a JSON pero se compone de una serie de *Points* que están situados con una latitud y una longitud específicas, que serán las del accidente; y pueden ser completados con información adicional. Para crear estos ficheros GeoJSON, OpenDataMadrid hace uso de su propio módulo de traducción de csv a GeoJSON.


#### Motor de ML ####

Una vez desplegada la aplicación, con todos los datos tratados y mostrados en el mapa es hora de hacer uso del Machine Learning. El usuario podrá clickar en cualquier punto del mapa y recibirá la probabilidad que existe de tener un accidente en ese punto y, en caso de producirse, con qué gravedad. El motor hace uso de un clasificador *RandomForest* con un acc promedio de 97%. 

Al ser un dataset relativamente pequeño y muy desbalanceado, hemos realizado un trabajo de *over-sampling* con el objetivo de mejorar el acc del clasificador, logrando con creces dicho objetivo. Este trabajo de *over-sampling* se realiza gracias a la libreria de python [imblearn](https://imbalanced-learn.readthedocs.io/en/stable/api.html).

# TODOs:
(El orden no indica prioridad)
1. Poner un único mapa, basado en HTML que permita contemplar tanto el clustering como los pins de las localizaciones
2. Filtrar la concentración y el número de pins por franja horaria
6. Cerciorarnos de que todo el dataset desde 2015 está bien integrado. Dejar ya limpio el dataset para reducir el tiempo de consulta o servirlo en vivo.
5. Pelearnos con Bootstrap para hacerlo más bonito.
