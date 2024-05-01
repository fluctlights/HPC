TARGETS = macqueen macqueen_OpenMP_V1 macqueen_OpenMP_V2

CC = g++
CFLAGS = -ggdb -std=c++17
CFLAGS_OpenMP = -ggdb -std=c++17 -fopenmp
LDFLAGS = `pkg-config opencv4 --cflags --libs`

SRC = macqueen.cpp
SRC_OpenMP_V1 = macqueen_OpenMP_V1.cpp
SRC_OpenMP_V2 = macqueen_OpenMP_V2.cpp

all: $(TARGETS)

macqueen: $(SRC_V1)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

macqueen_OpenMP_V1: $(SRC_OpenMP_V1)
	$(CC) $(CFLAGS_OpenMP) -o $@ $(SRC_OpenMP_V1) $(LDFLAGS)

macqueen_OpenMP_V2: $(SRC_OpenMP_V2)
	$(CC) $(CFLAGS_OpenMP) -o $@ $(SRC_OpenMP_V2) $(LDFLAGS)

clean:
	rm -f $(TARGETS)