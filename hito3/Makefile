TARGET = macqueen

SRC = macqueen.c

CC = gcc

LDFLAGS = -lm

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET) 2 0 100 0.2 1