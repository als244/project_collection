CC = gcc
CFLAGS = -g3 -std=c99 -pedantic -O2 -Wall
PROFILE_FLAGS = -pg -no-pie -fno-builtin

all: lstm_train

lstm_train: lstm.o hash_table.o
	${CC} ${CFLAGS} -o $@ $^ -lm ${PROFILE_FLAGS}

lstm.o: lstm.c
	${CC} ${CFLAGS} -c lstm.c

# hash_table_test: hash_table.o
# 	${CC} ${CFLAGS} -o $@ $^

hash_table.o: hash_table.c
	${CC} ${CFLAGS} -c hash_table.c

clean:
	rm *.o