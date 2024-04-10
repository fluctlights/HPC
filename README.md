# HPC

El repositorio contiene las siguientes carpetas:

- **hito2:** contiene el desarrollo correspondiente al Hito 2 del proyecto, que incluye el diseño de la solución paralela y un estudio completo de los tiempos de ejecución de la solución paralela propuesta, comparándolos con los tiempos registrados por la solución secuencial.

## Hito 2

- `requirements.txt`: contiene las dependencias y requisitos necesarios para el proyecto.
- `load.py`: carga el archivo `pavia.txt` para que se pueda trabajar con él.

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
