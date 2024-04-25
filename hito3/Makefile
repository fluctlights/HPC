TARGET = macqueen
SRC = macqueen.cpp
CC = g++
CFLAGS = -ggdb -std=c++17
LDFLAGS = `pkg-config opencv4 --cflags --libs`

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)