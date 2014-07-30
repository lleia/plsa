cc=gcc
cflags=-c -fpic -O2 -Wall 

model.o: model.c
	$(cc) $(cflags) model.c

training.o: training.c
	$(cc) $(cflags) training.c

parallel_training.o: parallel_training.c
	$(cc) $(cflags) parallel_training.c

foldin.o: foldin.c
	$(cc) $(cflags) foldin.c

plsa_suite.o: plsa_suite.c
	$(cc) $(cflags) plsa_suite.c

plsa: model.o training.o parallel_training.o foldin.o plsa_suite.o
	$(cc) -pthread model.o training.o parallel_training.o foldin.o plsa_suite.o -o plsa -lm

plsa.so: model.o training.o
	$(cc) -shared model.o training.o -o plsa.so -lm

plsa.a: model.o training.o
	ar rcs plsa.a model.o training.o parallel_training.o

all: plsa plsa.so plsa.a

clean:
	rm -rf *.o plsa plsa.so plsa.a
