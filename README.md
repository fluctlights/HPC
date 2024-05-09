# HPC

El repositorio contiene las siguientes carpetas:

- **hito2:** contiene el desarrollo correspondiente al Hito 2 del proyecto, que incluye el diseño de la solución paralela usando MPI y un estudio completo de los tiempos de ejecución de la solución paralela propuesta, comparándolos con los tiempos registrados por la solución secuencial.
- **hito3:** contiene el desarrollo correspondiente al Hito 3 del proyecto, que incluye el diseño de la solución paralela usando OpenMP y un estudio completo de los tiempos de ejecución de la solución paralela propuesta, comparándolos con los tiempos registrados por la solución secuencial.


## Hito 2

- `requirements.txt`: contiene las dependencias y requisitos necesarios para el proyecto.
- `load.py`: carga el archivo `pavia.txt` para que se pueda trabajar con él.
- `charts.py`: genera un gráfico con el tiempo de ejecución del directorio proporcionado.

Los algoritmos K-Means utilizados son:

- `MacQueen_V1.py`: version 1 secuencial.
- `MacQueen_V2.py`: version 2 secuencial.
- `MacQueenMPI_V1.py`: version 1 paralela usando MPI.
- `MacQueenMPI_V2.py`: version 2 paralela usando MPI.

Para finalizar, se incluye un archivo llamado `execute.sh` que realiza las siguientes acciones:

1. Configura un entorno virtual llamado `venv`.
2. Activa el entorno virtual.
3. Instala las dependencias y módulos necesarios especificados en el archivo `requirements.txt`.
4. Llama al archivo `load.py` para cargar el archivo `pavia.txt`.
5. llama a los diferentes algoritmos para registrar el tiempo de ejecución, y luego guarda los resultados en la carpeta `kmeans`, con un subdirectorio correspondiente a la versión utilizada.
6. Llama al archivo `charts.py` para generar gráficos con el tiempo de ejecución de las diferentes versiones en el subdirectorio correspondiente a la versión utilizada.

## Hito 3

- `load.py`: carga el archivo `pavia.txt` para que se pueda trabajar con él.
- `charts.py`: genera un gráfico con el tiempo de ejecución del directorio proporcionado.
- `Makefile`: archivo que contiene reglas que especifican cómo compilar el código fuente y cómo limpiar los archivos generados.
- `slurm.sh`: script utilizado para enviar trabajos en un clúster de HPC que utiliza SLURM.

Los algoritmos K-Means utilizados son:

- `macqueen.cpp`: version secuencial.
- `macqueen_OpenMP_V1.cpp`: version 1 paralela usando OpenMP.
- `macqueen_OpenMP_V2.cpp`: version 2 paralela usando OpenMP.

Para finalizar, se incluye un archivo llamado `execute.sh` que realiza las siguientes acciones:

1. Llama al archivo `load.py` para cargar el archivo `pavia.txt`.
2. Construye los ejecutables de los algoritmos K-Means.
3. llama a los diferentes algoritmos para registrar el tiempo de ejecución, y luego guarda los resultados en la carpeta `kmeans`, con un subdirectorio correspondiente a la versión utilizada.
4. Limpia los ejecutables generados utilizando _make clean_
5. Llama al archivo `charts.py` para generar gráficos con el tiempo de ejecución de las diferentes versiones en el subdirectorio correspondiente a la versión utilizada.
