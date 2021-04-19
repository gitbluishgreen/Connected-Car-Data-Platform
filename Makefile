objects = lex.yy.o Expression.tab.o kernel.o server.o car.o 
all: parser $(objects)
	nvcc -dlink -lrt -lpthread $(objects)

%.o: %.cu
	nvcc -dc $< -o $@ -I .

parser:
	yacc -Wall -d Expression.y -o Expression.tab.cpp
	mv Expression.tab.cpp Expression.tab.cu
	mv Expression.tab.hpp Expression.tab.cuh
	lex -o lex.yy.cu Expression.l

	
