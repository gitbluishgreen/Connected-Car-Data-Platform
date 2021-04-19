CFLAGS= -g -O3
LIBS= -lrt -lpthread
TARGET= app

all : $(TARGET)

$(TARGET) : parser server.o car.o proj_types.o kernel.o
	nvcc server.o car.o lex.yy.o Expression.tab.o proj_types.o kernel.o -o $(TARGET) $(CFLAGS) $(LIBS)

parser :    Expression.l Expression.y
	yacc -Wall -d Expression.y -o Expression.tab.cpp
	mv Expression.tab.cpp Expression.tab.cu
	mv Expression.tab.hpp Expression.tab.cuh
	lex -o lex.yy.cu Expression.l
	nvcc -c -dc Expression.tab.cu
	nvcc -c -dc lex.yy.cu

server.o : server.cu
	nvcc -c -dc server.cu $(CFLAGS) $(LIBS)

car.o : car.cu
	nvcc -c -dc car.cu $(CFLAGS) $(LIBS)

proj_types.o : proj_types.cu
	nvcc -c -dc proj_types.cu $(CFLAGS) $(LIBS)

kernel.o : kernel.cu
	nvcc -c -dc kernel.cu $(CFLAGS) $(LIBS)

clean :
	rm -f *.o
